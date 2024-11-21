#include <mitsuba/core/bsphere.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sunsky.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _emitter-constant:

Constant environment emitter (:monosp:`constant`)
-------------------------------------------------

.. pluginparameters::

 * - radiance
   - |spectrum|
   - Specifies the emitted radiance in units of power per unit area per unit steradian.
   - |exposed|, |differentiable|

This plugin implements a constant environment emitter, which surrounds
the scene and radiates diffuse illumination towards it. This is often
a good default light source when the goal is to visualize some loaded
geometry that uses basic (e.g. diffuse) materials.

.. tabs::
    .. code-tab:: xml
        :name: constant-light

        <emitter type="constant">
            <rgb name="radiance" value="1.0"/>
        </emitter>

    .. code-tab:: python

        'type': 'constant',
        'radiance': {
            'type': 'rgb',
            'value': 1.0,
        }

 */

#define NB_TURBIDITY 10
#define NB_ALBEDO 2
#define NB_CTRL_PTS 6
#define NB_PARAMS 9

#define NB_ETAS 30
#define NB_GAUSSIAN 5
#define NB_GAUSSIAN_PARAMS 5

#define DATABASE_PATH "sunsky-testing/res/datasets/"

template <typename Float, typename Spectrum>
class SunskyEmitter final : public Emitter<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Emitter, m_flags, m_to_world)
    MI_IMPORT_TYPES(Scene, Shape, Texture)

    using FloatStorage = DynamicBuffer<Float>;
    using SpecUInt32 = dr::uint32_array_t<Spectrum>;
    using SpecMask = dr::mask_t<Spectrum>;

    SunskyEmitter(const Properties &props) : Base(props) {
        /* Until `set_scene` is called, we have no information
           about the scene and default to the unit bounding sphere. */
        m_bsphere = BoundingSphere3f(ScalarPoint3f(0.f), 1.f);
        m_surface_area = 4.f * dr::Pi<Float>;

        // ================ GET TURBIDITY ===============
        ScalarFloat turbidity = props.get<ScalarFloat>("turbidity", 3.f);

        // ================= GET ALBEDO =================
        ref<Texture> albedo = props.texture<Texture>("albedo", 1.f);
        if (albedo->is_spatially_varying())
            Throw("Expected a non-spatially varying radiance spectra!");

        dr::eval(albedo);
        std::array<ScalarFloat, NB_CHANNELS> albedo_buff = extractAlbedo(albedo);

        // ================= GET ANGLES =================
        ScalarVector3f sun_dir = dr::normalize(
            m_to_world.scalar().inverse()
            .transform_affine(props.get<ScalarVector3f>("sun_direction")));

        ScalarPoint2f angles = from_scalar_spherical(sun_dir);
        angles.x() = dr::select(dr::sin(angles.x()) < 0, dr::TwoPi<Float> - angles.x(), angles.x());

        ScalarFloat sun_eta = 0.5f * dr::Pi<ScalarFloat> - angles.y();

        // ================= GET RADIANCE =================
        std::vector<ScalarFloat> dataset =
            array_from_file<double, ScalarFloat>(DATABASE_PATH + DATASET_NAME + ".bin");

        std::vector<ScalarFloat> rad_dataset =
            array_from_file<double, ScalarFloat>(DATABASE_PATH + DATASET_NAME + "_rad.bin");

        std::vector<ScalarFloat>
                params = getRadianceParams(dataset, albedo_buff, turbidity, sun_eta),
                radiance = getRadianceParams(rad_dataset, albedo_buff, turbidity, sun_eta);

        m_params = dr::load<FloatStorage>(params.data(), params.size());
        m_radiance = dr::load<FloatStorage>(radiance.data(), radiance.size());

        std::vector<ScalarFloat> tgmm_tables =
            array_from_file<float, ScalarFloat>(DATABASE_PATH "tgmm_tables.bin");

        auto [mis_weights, distrib_params] = getTGMMTables(tgmm_tables, turbidity, sun_eta);

        m_gaussians = dr::load<FloatStorage>(distrib_params.data(), distrib_params.size());
        m_gaussian_distr = DiscreteDistribution<Float>(mis_weights.data(), mis_weights.size());

        m_sun_dir = sun_dir;

        dr::make_opaque(m_params, m_radiance, m_gaussian_distr, m_gaussians, m_sun_dir);

        m_flags = +EmitterFlags::Infinite | +EmitterFlags::SpatiallyVarying;
    }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_object("radiance", m_temp_rad.get(), +ParamFlags::Differentiable);
    }

    void set_scene(const Scene *scene) override {
        if (scene->bbox().valid()) {
            ScalarBoundingSphere3f scene_sphere =
                scene->bbox().bounding_sphere();
            m_bsphere = BoundingSphere3f(scene_sphere.center, scene_sphere.radius);
            m_bsphere.radius =
                dr::maximum(math::RayEpsilon<Float>,
                        m_bsphere.radius * (1.f + math::RayEpsilon<Float>));
        } else {
            m_bsphere.center = 0.f;
            m_bsphere.radius = math::RayEpsilon<Float>;
        }
        m_surface_area = 4.f * dr::Pi<ScalarFloat> * dr::square(m_bsphere.radius);

        dr::make_opaque(m_bsphere.center, m_bsphere.radius, m_surface_area);
    }


    Spectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        Vector3f local_wi = dr::normalize(m_to_world.value().inverse().transform_affine(si.wi));
        Float cos_theta = Frame3f::cos_theta(-local_wi);
        Float cos_gamma = dr::dot(m_sun_dir, -local_wi);

        active &= cos_theta >= 0;

        if constexpr (is_rgb_v<Spectrum>) {

            SpecUInt32 idx = SpecUInt32({0, 1, 2});
            // dr::width(idx) == 1

            // Divide by normalisation factor
            return renderChannels(idx, cos_theta, cos_gamma, active) / 106.856980;

        } else {
            Spectrum normalized_wavelengths = (si.wavelengths - WAVELENGTHS[0]) / WAVELENGTH_STEP;
            SpecUInt32 query_idx = SpecUInt32(dr::floor(normalized_wavelengths));
            Spectrum lerp_factor = normalized_wavelengths - query_idx;

            SpecMask spec_mask = active & (query_idx >= 0) & (query_idx < 11);

            return dr::lerp(
                renderChannels(query_idx, cos_theta, cos_gamma, spec_mask),
                renderChannels(dr::minimum(query_idx + 1, NB_CHANNELS - 1), cos_theta, cos_gamma, spec_mask),
                lerp_factor);

        }
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &sample2, const Point2f &sample3,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        // 1. Sample spatial component
        Vector3f v0 = warp::square_to_uniform_sphere(sample2);
        Point3f orig = dr::fmadd(v0, m_bsphere.radius, m_bsphere.center);

        // 2. Sample diral component
        Vector3f v1 = warp::square_to_cosine_hemisphere(sample3),
                 dir = Frame3f(-v0).to_world(v1);

        // 3. Sample spectrum
        auto [wavelengths, weight] = sample_wavelengths(
            dr::zeros<SurfaceInteraction3f>(), wavelength_sample, active);

        weight *= m_surface_area * dr::Pi<ScalarFloat>;

        return { Ray3f(orig, dir, time, wavelengths),
                 depolarizer<Spectrum>(weight) };
    }

    std::pair<DirectionSample3f, Spectrum> sample_direction(const Interaction3f &it,
                                                            const Point2f &sample,
                                                            Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);

        Vector3f d = warp::square_to_uniform_sphere(sample);

        // Automatically enlarge the bounding sphere when it does not contain the reference point
        Float radius = dr::maximum(m_bsphere.radius, dr::norm(it.p - m_bsphere.center)),
              dist   = 2.f * radius;

        DirectionSample3f ds;
        ds.p       = dr::fmadd(d, dist, it.p);
        ds.n       = -d;
        ds.uv      = sample;
        ds.time    = it.time;
        ds.pdf     = warp::square_to_uniform_sphere_pdf(d);
        ds.delta   = false;
        ds.emitter = this;
        ds.d       = d;
        ds.dist    = dist;

        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        si.wavelengths = it.wavelengths;

        return {
            ds,
            depolarizer<Spectrum>(m_temp_rad->eval(si, active)) / ds.pdf
        };
    }

    Float pdf_direction(const Interaction3f &, const DirectionSample3f &ds,
                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        return warp::square_to_uniform_sphere_pdf(ds.d);
    }

    Spectrum eval_direction(const Interaction3f &it, const DirectionSample3f &,
                            Mask active) const override {
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        si.wavelengths = it.wavelengths;
        return depolarizer<Spectrum>(m_temp_rad->eval(si, active));
    }

    std::pair<Wavelength, Spectrum>
    sample_wavelengths(const SurfaceInteraction3f &si, Float sample,
                       Mask active) const override {
        return m_temp_rad->sample_spectrum(
            si, math::sample_shifted<Wavelength>(sample), active);
    }

    std::pair<PositionSample3f, Float>
    sample_position(Float /*time*/, const Point2f & /*sample*/,
                    Mask /*active*/) const override {
        if constexpr (dr::is_jit_v<Float>) {
            /* When virtual function calls are recorded in symbolic mode,
               we can't throw an exception here. */
            return { dr::zeros<PositionSample3f>(),
                     dr::full<Float>(dr::NaN<ScalarFloat>) };
        } else {
            NotImplementedError("sample_position");
        }
    }

    /// This emitter does not occupy any particular region of space, return an invalid bounding box
    ScalarBoundingBox3f bbox() const override {
        return ScalarBoundingBox3f();
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SunskyEmitter[" << std::endl
            << "  radiance = " << string::indent(m_temp_rad) << "," << std::endl
            << "  bsphere = " << string::indent(m_bsphere) << std::endl
            << "]";
        return oss.str();
    }


    MI_DECLARE_CLASS()
private:

    const std::string DATASET_NAME = is_spectral_v<Spectrum> ?
        "ssm_dataset_v2_spec" :
        "ssm_dataset_v2_rgb";

    static constexpr size_t NB_CHANNELS = is_spectral_v<Spectrum> ? 11 : 3,
                            DATASET_SIZE = NB_TURBIDITY * NB_ALBEDO * NB_CTRL_PTS * NB_CHANNELS * NB_PARAMS,
                            RAD_DATASET_SIZE = NB_TURBIDITY * NB_ALBEDO * NB_CTRL_PTS * NB_CHANNELS,
                            WAVELENGTH_STEP = 40;

    ref<Texture> m_temp_rad;

    Float m_surface_area;
    BoundingSphere3f m_bsphere;

    Float m_sun_phi;
    Vector3f m_sun_dir;

    FloatStorage m_radiance;
    FloatStorage m_params;
    FloatStorage m_gaussians;

    DiscreteDistribution<Float> m_gaussian_distr;


    static constexpr size_t WAVELENGTHS[11] = {
        320, 360, 400, 420, 460, 520, 560, 600, 640, 680, 720
    };


    Spectrum renderChannels(const SpecUInt32& idx,
        const Float& cos_theta, const Float& cos_gamma, const SpecMask& active) const {

        Spectrum coefs[NB_PARAMS];
        for (uint8_t i = 0; i < NB_PARAMS; ++i)
            coefs[i] = dr::gather<Spectrum>(m_params, idx * NB_PARAMS + i, active);

        Float gamma = dr::acos(cos_gamma);
        Float cos_gamma_sqr = dr::square(cos_gamma);

        Spectrum c1 = 1 + coefs[0] * dr::exp(coefs[1] / (cos_theta + 0.01f));
        Spectrum chi = (1 + cos_gamma_sqr) / dr::pow(1 + dr::square(coefs[8]) - 2 * coefs[8] * cos_gamma, 1.5);
        Spectrum c2 = coefs[2] + coefs[3] * dr::exp(coefs[4] * gamma) + coefs[5] * cos_gamma_sqr + coefs[6] * chi + coefs[7] * dr::safe_sqrt(cos_theta);

        return c1 * c2 * dr::gather<Spectrum>(m_radiance, idx, active);
    }

    MI_INLINE Point2f gaussianCdf(const Point2f& mu, const Point2f& sigma, const Point2f& x) const {
        return 0.5f * (1 + dr::erf(dr::InvSqrtTwo<Float> * (x - mu) / sigma));
    }

    Float tgmmPdf(const Vector3f& direction, Mask active) const {

        Point2f angles = from_spherical(direction);
        angles.x() += 0.5f * dr::Pi<ScalarFloat> - m_sun_phi;
        angles.x() = dr::select(angles.x() < 0, angles.x() + dr::TwoPi<ScalarFloat>, angles.x());

        active &= angles.y() >= 0 && angles.y() < 0.5f * dr::Pi<ScalarFloat>;

        const Point2f a = Point2f(0.0, 0.0),
                      b = Point2f(dr::TwoPi<Float>, 0.5f * dr::Pi<Float>);

        Float pdf = 0.0;
        for (size_t i = 0; i < 4 * NB_GAUSSIAN; ++i) {
            size_t baseIdx = i * NB_GAUSSIAN_PARAMS;

            Point2f mu = Point2f(m_gaussians[baseIdx + 0], m_gaussians[baseIdx + 1]),
                    sigma = Point2f(m_gaussians[baseIdx + 2], m_gaussians[baseIdx + 3]);

            Point2f cdf_a = gaussianCdf(mu, sigma, a),
                    cdf_b = gaussianCdf(mu, sigma, b);

            Point2f sample = (angles - mu) / sigma;

            Float gaussian_pdf = warp::square_to_std_normal_pdf(sample);
            gaussian_pdf *= m_gaussians[baseIdx + 4] / (cdf_b.x() - cdf_a.x()) * (cdf_b.y() - cdf_a.y()) * (sigma.x() * sigma.y());

            pdf += gaussian_pdf;
        }

        return dr::select(active, pdf, 0.0);
    }

    Vector3f tgmmSample(const Point2f& sample, const Mask& active) {
        UInt32 idx;
        Point2f sample_ = sample;
        std::tie(idx, sample_.x()) = m_gaussian_distr.sample_reuse(sample.x(), active);

        const Point2f a = Point2f(0.0, 0.0),
                      b = Point2f(dr::TwoPi<Float>, 0.5f * dr::Pi<Float>);

        return {};
    }

    std::vector<ScalarFloat> bezierInterpolate(
        const std::vector<ScalarFloat>& dataset, size_t outSize,
        const ScalarUInt32& offset, const ScalarFloat& x) const {

        constexpr ScalarFloat coefs[NB_CTRL_PTS] =
            {1, 5, 10, 10, 5, 1};

        std::vector<ScalarFloat> res(outSize, 0.0f);
        for (size_t i = 0; i < NB_CTRL_PTS; ++i) {
            ScalarFloat coef = coefs[i] * dr::pow(1 - x, 5 - i) * dr::pow(x, i);
            ScalarUInt32 index = offset + i * outSize;
            for (size_t j = 0; j < outSize; ++j)
                res[j] += coef * dataset[index + j];
        }

        return res;
    }

    static std::vector<ScalarFloat> lerp_vectors(const std::vector<ScalarFloat>& a, const std::vector<ScalarFloat>& b, const ScalarFloat& t) {
        assert(a.size() == b.size());

        std::vector<ScalarFloat> res(a.size());
        for (size_t i = 0; i < a.size(); ++i)
            res[i] = dr::lerp(a[i], b[i], t);

        return res;
    }

    auto getTGMMTables(const std::vector<ScalarFloat>& dataset,
                        ScalarFloat t, ScalarFloat eta) const {

        eta = dr::rad_to_deg(eta);
        ScalarFloat eta_idx_f = dr::clip((eta - 2) / 3, 0, NB_ETAS - 1),
                    t_idx_f = dr::clip(t - 2, 0, NB_TURBIDITY - 1);

        ScalarUInt32 eta_idx_low = dr::floor2int<ScalarUInt32>(eta_idx_f),
                     t_idx_low = dr::floor2int<ScalarUInt32>(t_idx_f);

        ScalarUInt32 eta_idx_high = dr::minimum(eta_idx_low + 1, NB_ETAS - 1),
                     t_idx_high = dr::minimum(t_idx_low + 1, (NB_TURBIDITY - 1) - 1);

        ScalarFloat eta_rem = eta_idx_f - eta_idx_low,
                    t_rem = t_idx_f - t_idx_low;

        const size_t tBlockSize = dataset.size() / (NB_TURBIDITY - 1),
                     etaBlockSize = tBlockSize / NB_ETAS;

        ScalarUInt64 indices[4] = {
            t_idx_low * tBlockSize + eta_idx_low * etaBlockSize,
            t_idx_low * tBlockSize + eta_idx_high * etaBlockSize,
            t_idx_high * tBlockSize + eta_idx_low * etaBlockSize,
            t_idx_high * tBlockSize + eta_idx_high * etaBlockSize
        };
        ScalarFloat lerpFactors[4] = {
            t_rem * eta_rem,
            t_rem * (1 - eta_rem),
            (1 - t_rem) * eta_rem,
            (1 - t_rem) * (1 - eta_rem)
        };
        std::vector<ScalarFloat> distribParams(4 * etaBlockSize);
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < etaBlockSize; ++j) {
                ScalarUInt32 index = i * etaBlockSize + j;
                distribParams[index] = dataset[indices[i] + j];
                distribParams[index] *= index % NB_GAUSSIAN_PARAMS == 4 ? lerpFactors[i] : 1;
            }
        }

        std::vector<ScalarFloat> misWeights(4 * NB_GAUSSIAN);
        for (size_t i = 0; i < 4 * NB_GAUSSIAN; ++i)
            misWeights[i] = dataset[i * (4 * NB_GAUSSIAN_PARAMS) + 4];

        return std::make_pair(misWeights, distribParams);
    }

    std::vector<ScalarFloat> getRadianceParams(
        const std::vector<ScalarFloat>& dataset,
        const std::array<ScalarFloat, NB_CHANNELS>& albedo,
        ScalarFloat turbidity, ScalarFloat eta) const {

        turbidity = dr::clip(turbidity, 1.f, NB_TURBIDITY);
        eta = dr::clip(eta, 0.f, 0.5f * dr::Pi<ScalarFloat>);

        ScalarFloat x = dr::pow(2 * dr::InvPi<ScalarFloat> * eta, 1.f/3.f);

        ScalarUInt32 t_int = dr::floor2int<ScalarUInt32>(turbidity),
               t_low = dr::maximum(t_int - 1, 0),
               t_high = dr::minimum(t_int, NB_TURBIDITY - 1);

        ScalarFloat t_rem = turbidity - t_int;

        const size_t tBlockSize = dataset.size() / NB_TURBIDITY,
                     aBlockSize = tBlockSize / NB_ALBEDO,
                     ctrlBlockSize = aBlockSize / NB_CTRL_PTS,
                     innerBlockSize = ctrlBlockSize / NB_CHANNELS;

        std::vector<ScalarFloat>
            t_low_a_low = bezierInterpolate(dataset, ctrlBlockSize, t_low * tBlockSize + 0 * aBlockSize, x),
            t_high_a_low = bezierInterpolate(dataset, ctrlBlockSize, t_high * tBlockSize + 0 * aBlockSize, x),
            t_low_a_high = bezierInterpolate(dataset, ctrlBlockSize, t_low * tBlockSize + 1 * aBlockSize, x),
            t_high_a_high = bezierInterpolate(dataset, ctrlBlockSize, t_high * tBlockSize + 1 * aBlockSize, x);

        std::vector<ScalarFloat>
            res_a_low = lerp_vectors(t_low_a_low, t_high_a_low, t_rem),
            res_a_high = lerp_vectors(t_low_a_high, t_high_a_high, t_rem);

        std::vector<ScalarFloat> res(ctrlBlockSize);
        for (size_t i = 0; i < ctrlBlockSize; ++i)
            res[i] = dr::lerp(res_a_low[i], res_a_high[i], albedo[i/innerBlockSize]);

        return res;
    }

    static auto extractAlbedo(const ref<Texture>& albedo) {
        std::array<ScalarFloat, NB_CHANNELS> albedo_buff = {};
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        if constexpr (!dr::is_array_v<Float> && is_rgb_v<Spectrum>) {
            Color3f temp = albedo->eval(si);
            for (size_t i = 0; i < NB_CHANNELS; ++i)
                albedo_buff[i] = temp[i];

        } else if constexpr (!dr::is_array_v<Float> && is_spectral_v<Spectrum>) {
            for (size_t i = 0; i < NB_CHANNELS; ++i) {
                si.wavelengths = WAVELENGTHS[i];
                albedo_buff[i] = albedo->eval(si)[0];
            }
        } else if constexpr (is_rgb_v<Spectrum>) {
            Color3f temp = albedo->eval(si);
            for (size_t i = 0; i < NB_CHANNELS; ++i)
                albedo_buff[i] = temp[i][0];
            Log(Info, "Result albedo: %f", temp);


        } else if constexpr (is_spectral_v<Spectrum>) {
            FloatStorage wavelengths = dr::load<size_t>(WAVELENGTHS, NB_CHANNELS);
            si.wavelengths = wavelengths;
            Spectrum res = albedo->eval(si);

            dr::eval(res);
            Float&& temp = dr::migrate(res[0], AllocType::Host);
            dr::sync_thread();

            for (size_t i = 0; i < NB_CHANNELS; ++i)
                albedo_buff[i] = temp[i];

        } else {
            Log(Error, "Unsupported spectrum type");
        }

        for (size_t i = 0; i < NB_CHANNELS; ++i)
            albedo_buff[i] = dr::clip(albedo_buff[i], 0.f, 1.f);

        return albedo_buff;
    }

    Point2f from_spherical(const Vector3f& v) const {
        return {
            dr::safe_sqrt(dr::atan2(v.y(), v.x())),
            drjit::unit_angle_z(v)
        };
    }

    ScalarPoint2f from_scalar_spherical(const ScalarVector3f& v) const {
        return {
            dr::safe_sqrt(dr::atan2(v.y(), v.x())),
            drjit::unit_angle_z(v)
        };
    }
};

MI_IMPLEMENT_CLASS_VARIANT(SunskyEmitter, Emitter)
MI_EXPORT_PLUGIN(SunskyEmitter, "Sun and Sky dome background emitter")
NAMESPACE_END(mitsuba)
