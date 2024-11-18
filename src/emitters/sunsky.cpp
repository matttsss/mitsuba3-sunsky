#include <mitsuba/core/bsphere.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/plugin.h>
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

    template<size_t N>
    using FloatArray = dr::Array<Float, N>;
    using DynamicArray = dr::DynamicArray<Float>;

    SunskyEmitter(const Properties &props) : Base(props) {
        /* Until `set_scene` is called, we have no information
           about the scene and default to the unit bounding sphere. */
        m_bsphere = BoundingSphere3f(ScalarPoint3f(0.f), 1.f);
        m_surface_area = 4.f * dr::Pi<Float>;

        Float turbidity = props.get<Float>("turbidity", 3.f);

        ref<Texture> albedo = props.texture<Texture>("albedo", 1.f);
        if (albedo->is_spatially_varying())
            Throw("Expected a non-spatially varying radiance spectra!");

        // EXTRACT ALBEDO PER WAVELENGTH / COLOR CHANEL
        SurfaceInteraction3f si;
        DynamicArray albedo_buff = dr::zeros<DynamicArray>(NB_CHANNELS);
        if constexpr (is_spectral_v<Spectrum>) {
            for (size_t i = 0; i < NB_CHANNELS; ++i) {
                si.wavelengths = Wavelength(WAVELENGTHS[i]);
                albedo_buff[i] = albedo->eval_1(si);
            }
        } else {
            Color3f temp = albedo->eval(si);
            for (size_t i = 0; i < NB_CHANNELS; ++i)
                albedo_buff[i] = temp[i];
        }

        Vector3f sun_dir = dr::normalize(
            m_to_world.value().inverse()
            .transform_affine(props.get<Vector3f>("sun_direction")));

        Point2f angles = from_spherical(sun_dir);
        angles.x() = dr::select(dr::sin(angles.x()) < 0, dr::TwoPi<Float> - angles.x(), angles.x());

        Float sun_eta = 0.5f * dr::Pi<Float> - angles.y();
        auto [unused1, dataset] =
            array_from_file_d<Float>(DATABASE_PATH + DATASET_NAME + ".bin");
        auto [unused2, rad_dataset] =
            array_from_file_d<Float>(DATABASE_PATH + DATASET_NAME + "_rad.bin");

        m_params = getRadianceParams<DATASET_SIZE>(dataset, albedo_buff, turbidity, sun_eta);
        m_radiance = getRadianceParams<RAD_DATASET_SIZE>(rad_dataset, albedo_buff, turbidity, sun_eta);


        auto [unused3, tgmm_tables] =
            array_from_file_f<Float>(DATABASE_PATH "tgmm_tables.bin");
        // TODO get weights and table

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

        return depolarizer<Spectrum>(m_temp_rad->eval(si, active));
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
protected:

    ref<Texture> m_temp_rad;
    BoundingSphere3f m_bsphere;

    /// Surface area of the bounding sphere
    Float m_surface_area;
    Float m_sun_phi;

    const std::string DATASET_NAME = is_spectral_v<Spectrum> ?
        "ssm_dataset_v2_spec" :
        "ssm_dataset_v2_rgb";

private:

    static constexpr size_t NB_CHANNELS = is_spectral_v<Spectrum> ? 11 : 3,
                            DATASET_SIZE = NB_TURBIDITY * NB_ALBEDO * NB_CTRL_PTS * NB_CHANNELS * NB_PARAMS,
                            RAD_DATASET_SIZE = NB_TURBIDITY * NB_ALBEDO * NB_CTRL_PTS * NB_CHANNELS,
                            WAVELENGTH_STEP = 40;

    FloatArray<NB_CHANNELS> m_radiance;
    FloatArray<NB_CHANNELS * NB_PARAMS> m_params;


    static constexpr size_t WAVELENGTHS[11] = {
        320, 360, 400, 420, 460, 520, 560, 600, 640, 680, 720
    };

    template<typename OutArray>
    OutArray bezierInterpolate(const DynamicArray& dataset,
        const UInt32& offset, const Float& x) {

        ScalarFloat coefs[NB_CTRL_PTS] = {1, 5, 10, 10, 5, 1};
        //UInt32 indices = offset + dr::arange<UInt32>(OutArray::Size);

        OutArray res = dr::zeros<OutArray>();
        for (size_t i = 0; i < NB_CTRL_PTS; ++i) {
            // FIXME use gather to avoid second loop
            //OutArray data = dr::gather<OutArray>(dataset, indices + i * OutArray::Size);
            //res += coefs[i] * data * dr::pow(1 - x, 5 - i) * dr::pow(x, i);
            for (size_t j = 0; j < OutArray::Size; ++j) {
                // FIXME use offset to get right index
                //res[j] += coefs[i] * dataset[offset + i * OutArray::Size + j] * dr::pow(1 - x, 5 - i) * dr::pow(x, i);
                res[j] += coefs[i] * dataset[0 + i * OutArray::Size + j] * dr::pow(1 - x, 5 - i) * dr::pow(x, i);
            }
        }

        return res;
    }

    template<size_t datasetSize>
    auto getRadianceParams(const DynamicArray& dataset,
        DynamicArray albedo, Float turbidity, Float eta) {

        turbidity = dr::clip(turbidity, 1.f, 10.f);
        albedo = dr::clip(albedo, 0.f, 1.f);

        eta = dr::clip(eta, 0.f, 0.5f * dr::Pi<Float>);
        Float x = dr::pow(2 * dr::InvPi<Float> * eta, 1.f/3.f);

        UInt32 t_int = dr::floor2int<UInt32>(turbidity),
               t_low = dr::maximum(t_int - 1, 0),
               t_high = dr::maximum(t_int - 1, 0);

        Float t_rem = turbidity - t_int;

        constexpr size_t tBlockSize = datasetSize / NB_TURBIDITY,
                         aBlockSize = tBlockSize / NB_ALBEDO,
                         ctrlBlockSize = aBlockSize / NB_CTRL_PTS,
                         innerBlockSize = ctrlBlockSize / NB_CHANNELS;

        using OutArray = FloatArray<ctrlBlockSize>;

        OutArray resALow = dr::lerp(
                    bezierInterpolate<OutArray>(dataset,
                        t_low * tBlockSize + 0 * aBlockSize, x),
                    bezierInterpolate<OutArray>(dataset,
                        t_high * tBlockSize + 0 * aBlockSize, x), t_rem);

        OutArray resAHigh = dr::lerp(
                    bezierInterpolate<OutArray>(dataset,
                        t_low * tBlockSize + 1 * aBlockSize, x),
                    bezierInterpolate<OutArray>(dataset,
                        t_high * tBlockSize + 1 * aBlockSize, x), t_rem);

        // FIXME manage to repeat the albedo array
        //OutArray res = dr::lerp(resALow, resAHigh, dr::repeat(albedo, innerBlockSize));
        OutArray res = dr::lerp(resALow, resAHigh, 0.5f);

        return res; // & (1 <= turbidity) & (turbidity <= 10) &
                     //(0 <= albedo) & (albedo <= 1) &
                     // (0 <= eta) & (eta <= 0.5f * dr::Pi<Float>);
    }

    Point2f from_spherical(const Vector3f& v) const {
        return Point2f(
            dr::safe_sqrt(dr::atan2(v.y(), v.x())),
            drjit::unit_angle_z(v)
        );
    }
};

MI_IMPLEMENT_CLASS_VARIANT(SunskyEmitter, Emitter)
MI_EXPORT_PLUGIN(SunskyEmitter, "Sun and Sky dome background emitter")
NAMESPACE_END(mitsuba)
