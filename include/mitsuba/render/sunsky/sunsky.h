#pragma once

#include <utility>
#include <vector>

#include <mitsuba/core/fstream.h>

// Used for creating the RGB solar dataset
#include <mitsuba/core/spectrum.h>

#include <drjit/dynamic.h>
#include <drjit/tensor.h>
#include <drjit/sphere.h>


NAMESPACE_BEGIN(mitsuba)

    /// Number of spectral channels in the skylight model
    #define NB_WAVELENGTHS 11
    /// Number of turbidity levels in the skylight model
    #define NB_TURBIDITY 10
	/// Number of albedo levels in the skylight model
    #define NB_ALBEDO 2

    /// Distance between wavelengths in the skylight model
    static constexpr size_t WAVELENGTH_STEP = 40;
    /// Wavelengths used in the skylight model
    template <typename Float>
    static constexpr Float WAVELENGTHS[NB_WAVELENGTHS] = {
        320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720
    };

    /// Number of control points for interpolation in the skylight model
    #define NB_SKY_CTRL_PTS 6
    /// Number of parameters for the skylight model equation
    #define NB_SKY_PARAMS 9

    /// Number of control points for interpolation in the sun model
    #define NB_SUN_CTRL_PTS 4
    /// Number of segments for the piecewise polynomial in the sun model
    #define NB_SUN_SEGMENTS 45
    /// Number of coefficients for the sun's limb darkening
    #define NB_SUN_LD_PARAMS 6

    /// Number of elevation control points for the tgmm sampling tables
    #define NB_ETAS 30
    /// Number of gaussian components in the tgmm
    #define NB_GAUSSIAN 5
    /// Number of parameters for each gaussian component
    #define NB_GAUSSIAN_PARAMS 5

    /// Sun aperture angle in degrees
    #define SUN_APERTURE 0.5358
    /// Mean radius of the Earth
    #define EARTH_MEAN_RADIUS 6371.01   // In km
    /// Astronomical unit
    #define ASTRONOMICAL_UNIT 149597890 // In km


    // ================================================================================================
    // ====================================== HELPER FUNCTIONS ========================================
    // ================================================================================================

    template<typename Value>
    Vector<Value, 3> to_spherical(const Point<Value, 2>& angles) {
        auto [sp, cp] = dr::sincos(angles.x());
        auto [st, ct] = dr::sincos(angles.y());

        return {
            cp * st, sp * st, ct
        };
    }

    template<typename Value>
    Point<Value, 2> from_spherical(const Vector<Value, 3>& v) {
        return {
            dr::atan2(v.y(), v.x()),
            dr::unit_angle_z(v)
        };
    }


    // ================================================================================================
    // ========================================== SKY MODEL ===========================================
    // ================================================================================================

    /**
     * \brief Interpolates the dataset along a quintic bezier curve
     *
     * @param dataset Dataset to interpolate using a degree 5 bezier curve
     * @param out_size Size of the data to interpolate
     * @param offset Offset to apply to the indices
     * @param x Interpolation factor
     * @return Interpolated data
     */
    template <typename Float>
    DynamicBuffer<Float> bezier_interpolate(
        const DynamicBuffer<Float>& dataset, const uint32_t out_size,
        const dr::uint32_array_t<Float>& offset, const Float& x) {

        using ScalarFloat = dr::value_t<Float>;
        using FloatStorage = DynamicBuffer<Float>;
        using UInt32Storage = DynamicBuffer<dr::uint32_array_t<Float>>;

        UInt32Storage indices = offset + dr::arange<UInt32Storage>(out_size);
        constexpr ScalarFloat coefs[NB_SKY_CTRL_PTS] = {1, 5, 10, 10, 5, 1};

        FloatStorage res = dr::zeros<FloatStorage>(out_size);
        for (uint8_t ctrl_pt = 0; ctrl_pt < NB_SKY_CTRL_PTS; ++ctrl_pt) {
            FloatStorage data = dr::gather<FloatStorage>(dataset, indices + ctrl_pt * out_size);
            res += coefs[ctrl_pt] * dr::pow(x, ctrl_pt) * dr::pow(1 - x, (NB_SKY_CTRL_PTS - 1) - ctrl_pt) * data;
        }

        return res;
    }

    /**
     * \brief Pre-computes the sky dataset using turbidity, albedo and sun
     * elevation
     *
     * @tparam dataset_size Size of the dataset
     * @param dataset Dataset to interpolate
     * @param albedo Albedo values corresponding to each channel
     * @param turbidity Turbidity used for the skylight model
     * @param eta Sun elevation angle
     * @return The interpolated dataset
     */
    template <uint32_t dataset_size, typename Float>
    DynamicBuffer<Float> compute_radiance_params(
        const DynamicBuffer<Float>& dataset,
        const DynamicBuffer<Float>& albedo, Float turbidity, Float eta) {

        using UInt32 = dr::uint32_array_t<Float>;
        using UInt32Storage = DynamicBuffer<UInt32>;
        using FloatStorage = DynamicBuffer<Float>;

        // Clip parameters to valid ranges
        turbidity = dr::clip(turbidity, 1.f, 10.f);
        eta = dr::clip(eta, 0.f, 0.5f * dr::Pi<Float>);

        Float x = dr::pow(2 * dr::InvPi<Float> * eta, 1.f/3.f);

        UInt32 t_int = dr::floor2int<UInt32>(turbidity),
               t_low = dr::maximum(t_int - 1, 0),
               t_high = dr::minimum(t_low + 1, NB_TURBIDITY - 1);

        // TODO gradient breaks here
        Float t_rem = turbidity - t_int;

        // Compute block sizes for each parameter to facilitate indexing
        const uint32_t t_block_size = dataset_size / NB_TURBIDITY,
                       a_block_size = t_block_size  / NB_ALBEDO,
                       result_size  = a_block_size / NB_SKY_CTRL_PTS,
                       nb_params    = result_size / albedo.size();
                                                 // albedo.size() <==> NB_CHANNELS

        // Interpolate on elevation
        FloatStorage
            t_low_a_low = bezier_interpolate(dataset, result_size, t_low * t_block_size + 0 * a_block_size, x),
            t_high_a_low = bezier_interpolate(dataset, result_size, t_high * t_block_size + 0 * a_block_size, x),
            t_low_a_high = bezier_interpolate(dataset, result_size, t_low * t_block_size + 1 * a_block_size, x),
            t_high_a_high = bezier_interpolate(dataset, result_size, t_high * t_block_size + 1 * a_block_size, x);

        // Interpolate on turbidity
        FloatStorage res_a_low = dr::lerp(t_low_a_low, t_high_a_low, t_rem),
                     res_a_high = dr::lerp(t_low_a_high, t_high_a_high, t_rem);

        // Interpolate on albedo
        UInt32Storage idx = dr::arange<UInt32Storage>(nb_params * albedo.size());
        idx /= nb_params;

        return dr::lerp(res_a_low, res_a_high, dr::gather<FloatStorage>(albedo, idx));
    }


    // ================================================================================================
    // ========================================== SUN MODEL ===========================================
    // ================================================================================================

    template<typename Float>
    struct DateTimeRecord {
        int year;
        int month;
        int day;
        Float hour;
        Float minute;
        Float second;

        std::string toString() const {
            std::ostringstream oss;
            oss << "DateTimeRecord[year = " << year
                << ", month= " << month
                << ", day = " << day
                << ", hour = " << hour
                << ", minute = " << minute
                << ", second = " << second << "]";
            return oss.str();
        }
    };

    template<typename Float>
    struct LocationRecord {
        Float longitude;
        Float latitude;
        Float timezone;

        std::string toString() const {
            std::ostringstream oss;
            oss << "LocationRecord[latitude = " << latitude
                << ", longitude = " << longitude
                << ", timezone = " << timezone << "]";
            return oss.str();
        }
    };

    /**
     * \brief Compute the elevation and azimuth of the sun as seen by an observer
     * at \c location at the date and time specified in \c dateTime.
     *
     * Based on "Computing the Solar Vector" by Manuel Blanco-Muriel,
     * Diego C. Alarcon-Padilla, Teodoro Lopez-Moratalla, and Martin Lara-Coira,
     * in "Solar energy", vol 27, number 5, 2001 by Pergamon Press.
     */
    template <typename Float>
    Vector<Float, 3> compute_sun_coordinates(const DateTimeRecord<Float>& dateTime, const LocationRecord<Float>& location) {
        using Int32 = dr::int32_array_t<Float>;

        // Main variables
        Float elapsedJulianDays, decHours;
        Float eclipticLongitude, eclipticObliquity;
        Float rightAscension, declination;
        Float elevation, azimuth;

        // Auxiliary variables
        Float dY;
        Float dX;

        /* Calculate difference in days between the current Julian Day
           and JD 2451545.0, which is noon 1 January 2000 Universal Time */
        {
            // Calculate time of the day in UT decimal hours
            decHours = dateTime.hour - location.timezone +
                (dateTime.minute + dateTime.second / 60.0 ) / 60.0;

            // Calculate current Julian Day
            Int32 liAux1 = (dateTime.month-14) / 12;
            Int32 liAux2 = (1461*(dateTime.year + 4800 + liAux1)) / 4
                + (367 * (dateTime.month - 2 - 12 * liAux1)) / 12
                - (3 * ((dateTime.year + 4900 + liAux1) / 100)) / 4
                + dateTime.day - 32075;
            Float dJulianDate = liAux2 - 0.5 + decHours / 24.0;

            // Calculate difference between current Julian Day and JD 2451545.0
            elapsedJulianDays = dJulianDate - 2451545.0;
        }

        /* Calculate ecliptic coordinates (ecliptic longitude and obliquity of the
           ecliptic in radians but without limiting the angle to be less than 2*Pi
           (i.e., the result may be greater than 2*Pi) */
        {
            Float omega = 2.1429 - 0.0010394594 * elapsedJulianDays;
            Float meanLongitude = 4.8950630 + 0.017202791698 * elapsedJulianDays; // Radians
            Float anomaly = 6.2400600 + 0.0172019699 * elapsedJulianDays;

            eclipticLongitude = meanLongitude + 0.03341607 * dr::sin(anomaly)
                + 0.00034894 * dr::sin(2*anomaly) - 0.0001134
                - 0.0000203 * dr::sin(omega);

            eclipticObliquity = 0.4090928 - 6.2140e-9 * elapsedJulianDays
                + 0.0000396 * dr::cos(omega);
        }

        /* Calculate celestial coordinates ( right ascension and declination ) in radians
           but without limiting the angle to be less than 2*Pi (i.e., the result may be
           greater than 2*Pi) */
        {
            Float sinEclipticLongitude = dr::sin(eclipticLongitude);
            dY = dr::cos(eclipticObliquity) * sinEclipticLongitude;
            dX = dr::cos(eclipticLongitude);
            rightAscension = dr::atan2(dY, dX);
            rightAscension += dr::select(rightAscension < 0.0, dr::TwoPi<Float>, 0.0);

            declination = dr::asin(dr::sin(eclipticObliquity) * sinEclipticLongitude);
        }

        // Calculate local coordinates (azimuth and zenith angle) in degrees
        {
            Float greenwichMeanSiderealTime = 6.6974243242
                + 0.0657098283 * elapsedJulianDays + decHours;

            Float localMeanSiderealTime = degToRad(greenwichMeanSiderealTime * 15
                + location.longitude);

            Float latitudeInRadians = degToRad(location.latitude);
            Float cosLatitude = dr::cos(latitudeInRadians);
            Float sinLatitude = dr::sin(latitudeInRadians);

            Float hourAngle = localMeanSiderealTime - rightAscension;
            Float cosHourAngle = dr::cos(hourAngle);

            elevation = dr::acos(cosLatitude * cosHourAngle
                * dr::cos(declination) + dr::sin(declination) * sinLatitude);

            dY = -dr::sin(hourAngle);
            dX = dr::tan(declination) * cosLatitude - sinLatitude * cosHourAngle;

            azimuth = dr::atan2(dY, dX);
            azimuth += dr::select(azimuth < 0.0, dr::TwoPi<Float>, 0.0);

            // Parallax Correction
            elevation += (EARTH_MEAN_RADIUS / ASTRONOMICAL_UNIT) * dr::sin(elevation);
        }

        return from_spherical(Point<Float, 2>(azimuth, elevation));
    }

    /**
     * \brief Collects and linearly interpolates the sun radiance dataset along
     * turbidity
     *
     * @tparam dataset_size Size of the dataset
     * @param sun_radiance_dataset Dataset to interpolate
     * @param turbidity Turbidity used for the skylight model
     * @return The interpolated dataset
     */
    template <uint32_t dataset_size, typename Float>
    DynamicBuffer<Float> compute_sun_params(const DynamicBuffer<Float>& sun_radiance_dataset, Float turbidity) {
        using UInt32 = dr::uint32_array_t<Float>;
        using UInt32Storage = DynamicBuffer<dr::uint32_array_t<Float>>;
        using FloatStorage = DynamicBuffer<Float>;

        turbidity = dr::clip(turbidity - 1, 0.f, 9.f);
        UInt32 t_low = dr::floor2int<UInt32>(turbidity),
               t_high = t_low + 1;
        Float  t_rem = turbidity - t_low;

        constexpr uint32_t t_block_size = dataset_size / NB_TURBIDITY;

        UInt32Storage idx = dr::arange<UInt32Storage>(t_block_size);
        return dr::lerp(dr::gather<FloatStorage>(sun_radiance_dataset, t_low * t_block_size + idx, t_low < NB_TURBIDITY - 1),
                        dr::gather<FloatStorage>(sun_radiance_dataset, t_high * t_block_size + idx, t_high < NB_TURBIDITY - 1),
                        t_rem);
    }

    // ================================================================================================
    // ======================================== SAMPLING MODEL ========================================
    // ================================================================================================

    /**
     * \brief Extracts the Gaussian Mixture Model parameters from the TGMM dataset
     * The 4 * (5 gaussians) cannot be interpolated directly, so we need to
     * combine them and adjust the weights based on the elevation and turbidity
     * linear interpolation parameters.
     *
     * @tparam dataset_size Size of the TGMM dataset
     * @param tgmm_tables Dataset for the Gaussian Mixture Models
     * @param turbidity Turbidity used for the skylight model
     * @param eta Elevation of the sun
     * @return The new distribution parameters and the mixture weights
     */
    template <uint32_t dataset_size, typename Float>
    auto compute_tgmm_distribution(const DynamicBuffer<Float>& tgmm_tables, Float turbidity, Float eta) {
        using UInt32 = dr::uint32_array_t<Float>;
        using ScalarUInt32 = dr::value_t<UInt32>;
        using UInt32Storage = DynamicBuffer<UInt32>;
        using FloatStorage = DynamicBuffer<Float>;

        // ============== EXTRACT INDEXES AND LERP WEIGHTS ==============

        eta = dr::rad_to_deg(eta);
        Float eta_idx_f = dr::clip((eta - 2) / 3, 0, NB_ETAS - 1),
              t_idx_f = dr::clip(turbidity - 2, 0, (NB_TURBIDITY - 1) - 1);

        UInt32 eta_idx_low = dr::floor2int<UInt32>(eta_idx_f),
               t_idx_low = dr::floor2int<UInt32>(t_idx_f);

        UInt32 eta_idx_high = dr::minimum(eta_idx_low + 1, NB_ETAS - 1),
               t_idx_high = dr::minimum(t_idx_low + 1, (NB_TURBIDITY - 1) - 1);

        Float eta_rem = eta_idx_f - eta_idx_low,
              t_rem = t_idx_f - t_idx_low;

        constexpr uint32_t t_block_size = dataset_size / (NB_TURBIDITY - 1),
                           result_size = t_block_size / NB_ETAS;

        const UInt32 indices[4] = {
            t_idx_low * t_block_size + eta_idx_low * result_size,
            t_idx_low * t_block_size + eta_idx_high * result_size,
            t_idx_high * t_block_size + eta_idx_low * result_size,
            t_idx_high * t_block_size + eta_idx_high * result_size
        };

        const Float lerp_factors[4] = {
            (1 - t_rem) * (1 - eta_rem),
            (1 - t_rem) * eta_rem,
            t_rem * (1 - eta_rem),
            t_rem * eta_rem
        };

        // ==================== EXTRACT PARAMETERS AND APPLY LERP WEIGHT ====================
        FloatStorage distrib_params = dr::zeros<FloatStorage>(4 * result_size);
        UInt32Storage param_indices = dr::arange<UInt32Storage>(result_size);
        for (ScalarUInt32 mixture_idx = 0; mixture_idx < 4; ++mixture_idx) {

            // Gather gaussian weights and parameters
            FloatStorage params = dr::gather<FloatStorage>(tgmm_tables, indices[mixture_idx] + param_indices);

            // Apply lerp factor to gaussian weights
            //dr::scatter_reduce(ReduceOp::Mul, params, (FloatStorage)lerp_factors[mixture_idx], gaussian_weight_idx);
            for (ScalarUInt32 weight_idx = 0; weight_idx < NB_GAUSSIAN; ++weight_idx) {
                ScalarUInt32 gaussian_weight_idx = weight_idx * NB_GAUSSIAN_PARAMS + (NB_GAUSSIAN_PARAMS - 1);
                dr::scatter(params, params[gaussian_weight_idx] * lerp_factors[mixture_idx], (UInt32)gaussian_weight_idx);
            }

            // Scatter back the parameters in the final gaussian mixture buffer
            dr::scatter(distrib_params, params, mixture_idx * result_size + param_indices);

        }

        // Extract gaussian weights
        UInt32Storage mis_weight_idx = NB_GAUSSIAN_PARAMS * dr::arange<UInt32Storage>(4 * NB_GAUSSIAN) + (NB_GAUSSIAN_PARAMS - 1);
        FloatStorage mis_weights = dr::gather<FloatStorage>(distrib_params, mis_weight_idx);

        return std::make_pair(distrib_params, mis_weights);
    }

    // ================================================================================================
    // =========================================== FILE I/O ===========================================
    // ================================================================================================

    /**
     * \brief Extracts an array from a compatible file
     *
     * @tparam FileType Type of the data stored in the file
     * @tparam OutType The type of the data to return
     * @param path Path of the data file
     * @return The extracted array
     */
    template <typename FileType, typename OutType>
    DynamicBuffer<OutType> array_from_file(const std::string &path) {
        FileStream file(path, FileStream::EMode::ERead);

        using ScalarFileType = dr::value_t<FileType>;
        using FileStorage = DynamicBuffer<FileType>;
        using FloatStorage = DynamicBuffer<OutType>;

        // =============== Read headers ===============
        char text_buff[5] = "";
        file.read(text_buff, 3);
        if (strcmp(text_buff, "SKY") != 0 && strcmp(text_buff, "SUN") != 0)
            Throw("OUPSSS wrong file");

        // Read version
        uint32_t version;
        file.read(version);

        // =============== Read tensor dimensions ===============
        size_t nb_dims = 0;
        file.read(nb_dims);

        size_t nb_elements = 1;
        std::vector<size_t> shape(nb_dims);
        for (size_t dim = 0; dim < nb_dims; ++dim) {
            file.read(shape[dim]);

            if (!shape[dim])
                Throw("Got dimension with 0 elements");

            nb_elements *= shape[dim];
        }

        // ==================== Read data ====================
        std::vector<ScalarFileType> buffer(nb_elements);
        file.read_array(buffer.data(), nb_elements);
        file.close();

        FileStorage data_f = dr::load<FileStorage>(buffer.data(), nb_elements);

        return FloatStorage(data_f);
    }

    /**
    * \brief Writes a DynamicBuffer to a file with a specific format compatible
    * with the sunsky model
    *
    * @tparam Float Type of the data to store in the file
    * @param path Path of the data file
    * @param data Data to store in the file
    * @param shape Shape of the data to store (optional)
    */
    template<typename Float>
    void array_to_file(const std::string &path, const DynamicBuffer<Float>& data, std::vector<size_t> shape = {}) {
        if (shape.empty()) shape.push_back(data.size());

        FileStream file(path, FileStream::EMode::ETruncReadWrite);

        // =============== Write headers ===============
        // Write headers
        file.write("SKY", 3);
        file.write((uint32_t) 0);

        // =============== Write dimensions ===============
        file.write((size_t) shape.size());

        for (size_t dim_length : shape) {
            file.write(dim_length);

            if (!dim_length)
                Throw("Got dimension with 0 elements");
        }

        // ==================== Write data ====================
        file.write_array(data.data(), data.size());

        file.close();
    }


    /*
     * This section contains the code that was used to generate the dataset files
     * from the original header files. These functions may not look friendly, but
     * they mainly reorder the data by swapping axis. They should only be used if
     * the generated dataset files are lost.
     * The exception being the Solar RGB dataset that need to be computed via the
     * Solar spectral dataset and the limb darkening dataset.
     */

    // Header with datasets downloadable from here:
    // https://cgg.mff.cuni.cz/projects/SkylightModelling/
    #include "ArHosekSkyModelData_Spectral.h"
    #include "ArHosekSkyModelData_RGB.h"
    #include "ArHosekSkyModelData_CIEXYZ.h"

    #define F_DIM 5
    #define L_DIM 4

    enum class SunDataShapeIdx: uint32_t { WAVELENGTH = 0, TURBIDITY, SUN_SEGMENTS, SUN_CTRL_PTS };
    enum class SkyDataShapeIdx: uint32_t { WAVELENGTH = 0, ALBEDO, TURBIDITY, CTRL_PT, PARAMS };

    // =========================== SHAPES OF THE ORIGINAL WILKIE-HOSEK DATASETS ============================
    constexpr size_t solar_shape[4] = {NB_WAVELENGTHS, NB_TURBIDITY, NB_SUN_SEGMENTS, NB_SUN_CTRL_PTS};
    constexpr size_t limb_darkening_shape[2] = {NB_WAVELENGTHS, 6};

    constexpr size_t f_spec_shape[F_DIM] = {NB_WAVELENGTHS, NB_ALBEDO, NB_TURBIDITY, NB_SKY_CTRL_PTS, NB_SKY_PARAMS};
    constexpr size_t l_spec_shape[L_DIM] = {NB_WAVELENGTHS, NB_ALBEDO, NB_TURBIDITY, NB_SKY_CTRL_PTS};

    constexpr size_t f_tri_shape[F_DIM] = {3, NB_ALBEDO, NB_TURBIDITY, NB_SKY_CTRL_PTS, NB_SKY_PARAMS};
    constexpr size_t l_tri_shape[L_DIM] = {3, NB_ALBEDO, NB_TURBIDITY, NB_SKY_CTRL_PTS};

    /// Helper struct to link the dataset metadata
    struct Dataset {
        size_t nb_dims;
        const size_t* dim_size;

        double** dataset;
    };

    constexpr Dataset f_spectral = {
        .nb_dims = F_DIM,
        .dim_size = f_spec_shape,
        .dataset = datasets
    };

    constexpr Dataset l_spectral = {
        .nb_dims = L_DIM,
        .dim_size = l_spec_shape,
        .dataset = datasetsRad
    };

    constexpr Dataset f_RGB = {
        .nb_dims = F_DIM,
        .dim_size = f_tri_shape,
        .dataset = datasetsRGB
    };

    constexpr Dataset l_RGB = {
        .nb_dims = L_DIM,
        .dim_size = l_tri_shape,
        .dataset = datasetsRGBRad
    };

    constexpr Dataset f_XYZ = {
        .nb_dims = F_DIM,
        .dim_size = f_tri_shape,
        .dataset = datasetsXYZ
    };

    constexpr Dataset l_XYZ = {
        .nb_dims = L_DIM,
        .dim_size = l_tri_shape,
        .dataset = datasetsXYZRad
    };

    constexpr Dataset solar_dataset = {
        .nb_dims = 4,
        .dim_size = solar_shape,
        .dataset = solarDatasets
    };

    constexpr Dataset limb_darkening_dataset = {
        .nb_dims = 2,
        .dim_size = limb_darkening_shape,
        .dataset = limbDarkeningDatasets
    };

    /**
     * Serializes the Limb Darkening dataset to a file
     *
     * @param path Path of the file to write to
     */
    void write_limb_darkening_data(const std::string& path) {
        const auto [nb_dims, dim_size, p_dataset] = limb_darkening_dataset;
        FileStream file(path, FileStream::EMode::ETruncReadWrite);

        // Write headers
        file.write("SUN", 3);
        file.write((uint32_t)0);

        // Write tensor dimensions
        file.write(nb_dims);

        // Write shapes as [wavelengths x nb_params]
        file.write(dim_size[0]);
        file.write(dim_size[1]);

        // Flatten the data to the file
        for (size_t w = 0; w < NB_WAVELENGTHS; ++w)
            file.write_array(p_dataset[w], NB_SUN_LD_PARAMS);

        file.close();
    }

    /**
     * Precomputes the sun RGB dataset from the spectral dataset and
     * the limb darkening dataset
     *
     * @param path Path of the file to write to
     */
    void write_sun_data_rgb(const std::string& path) {
        const auto [nb_dims_solar, dim_size_solar, p_dataset_solar] = solar_dataset;
        const auto [nb_dims_ld, dim_size_ld, p_dataset_ld] = limb_darkening_dataset;
        FileStream file(path, FileStream::EMode::ETruncReadWrite);

        // Write headers
        file.write("SUN", 3);
        file.write((uint32_t)0);

        // Write tensor dimensions
        file.write((size_t) 5);

        // Write reordered shapes
        file.write((size_t) NB_TURBIDITY);
        file.write((size_t) NB_SUN_SEGMENTS);
        file.write((size_t) 3); // RGB channels
        file.write((size_t) NB_SUN_CTRL_PTS);
        file.write((size_t) NB_SUN_LD_PARAMS);

        size_t dst_idx = 0;
        double* buffer = (double*)calloc(NB_TURBIDITY * NB_SUN_SEGMENTS * 3 * NB_SUN_CTRL_PTS * NB_SUN_LD_PARAMS, sizeof(double));

        for (size_t turb = 0; turb < NB_TURBIDITY; ++turb) {

            for (size_t segment = 0; segment < NB_SUN_SEGMENTS; ++segment) {

                for (size_t rgb_idx = 0; rgb_idx < 3; ++rgb_idx) {

                    for (size_t ctrl_pt = 0; ctrl_pt < NB_SUN_CTRL_PTS; ++ctrl_pt) {
                        // Weird indices since their dataset goes backwards on the last index
                        const size_t sun_idx = turb * (NB_SUN_SEGMENTS * NB_SUN_CTRL_PTS) +
                                                      (segment + 1) * NB_SUN_CTRL_PTS - (ctrl_pt + 1);

                        for (size_t ld_param_idx = 0; ld_param_idx < NB_SUN_LD_PARAMS; ++ld_param_idx) {

                            // Convert from spectral to RGB
                            for (size_t lambda = 0; lambda < NB_WAVELENGTHS; ++lambda) {
                                const double rectifier = linear_rgb_rec(WAVELENGTHS<double>[lambda])[rgb_idx];

                                buffer[dst_idx] += rectifier * p_dataset_solar[lambda][sun_idx] *
                                                        p_dataset_ld[lambda][ld_param_idx];
                            }

                            buffer[dst_idx] /= NB_WAVELENGTHS;
                            ++dst_idx;
                        }
                    }
                }
            }
        }

        file.write_array(buffer, NB_TURBIDITY * NB_SUN_SEGMENTS * 3 * NB_SUN_CTRL_PTS * NB_SUN_LD_PARAMS);
        file.close();
        free(buffer);
    }

    /**
     * Re-orders the sun spectral dataset from the original dataset
     * From [wavelengths x turbidity x sun_segments x sun_ctrl_pts]
     * to   [turbidity x sun_segments x wavelengths x sun_ctrl_pts]
     *
     * @param path Path of the file to write to
     */
    void write_sun_data_spectral(const std::string& path) {
        const auto [nb_dims, dim_size, p_dataset] = solar_dataset;
        FileStream file(path, FileStream::EMode::ETruncReadWrite);

        // Write headers
        file.write("SUN", 3);
        file.write((uint32_t)0);

        // Write tensor dimensions
        file.write(nb_dims);

        // Write reordered shapes
        file.write(dim_size[(uint32_t)SunDataShapeIdx::TURBIDITY]);
        file.write(dim_size[(uint32_t)SunDataShapeIdx::SUN_SEGMENTS]);
        file.write(dim_size[(uint32_t)SunDataShapeIdx::WAVELENGTH]);
        file.write(dim_size[(uint32_t)SunDataShapeIdx::SUN_CTRL_PTS]);

        for (size_t turb = 0; turb < NB_TURBIDITY; ++turb) {
            for (size_t segment = 0; segment < NB_SUN_SEGMENTS; ++segment) {
                for (size_t lambda = 0; lambda < NB_WAVELENGTHS; ++lambda) {
                    for (size_t ctrl_pt = 0; ctrl_pt < NB_SUN_CTRL_PTS; ++ctrl_pt) {
                        // Weird indices since their dataset goes backwards on the last index
                        const size_t src_global_offset = turb * (NB_SUN_SEGMENTS * NB_SUN_CTRL_PTS) +
                                                         (segment + 1) * NB_SUN_CTRL_PTS - (ctrl_pt + 1);

                        file.write(p_dataset[lambda][src_global_offset]);
                    }
                }
            }
        }

        file.close();
    }


    /**
     * Re-orders the sky dataset from the original dataset
     * From [wavelengths/channels x albedo x turbidity x sky_ctrl_pts (x sky_params)]
     * to   [turbidity x albedo x sky_ctrl_pts x wavelengths/channels (x sky_params)]
     *
     * @param path Path of the file to write to
     */
    void write_sky_data(const std::string &path, const Dataset& dataset) {
        const auto [nb_dims, dim_size, p_dataset] = dataset;
        FileStream file(path, FileStream::EMode::ETruncReadWrite);

        // Write headers
        file.write("SKY", 3);
        file.write((uint32_t)0);

        // Write tensor dimensions
        file.write(nb_dims);

        size_t tensor_size = 1;
        for (size_t dim = 0; dim < nb_dims; ++dim)
            tensor_size *= dim_size[dim];


        // Write reordered shapes
        file.write(dim_size[(uint32_t)SkyDataShapeIdx::TURBIDITY]);
        file.write(dim_size[(uint32_t)SkyDataShapeIdx::ALBEDO]);
        file.write(dim_size[(uint32_t)SkyDataShapeIdx::CTRL_PT]);
        file.write(dim_size[(uint32_t)SkyDataShapeIdx::WAVELENGTH]);

        if (nb_dims == F_DIM)
            file.write(dim_size[(uint32_t)SkyDataShapeIdx::PARAMS]);


        const size_t nb_params = nb_dims == F_DIM ? NB_SKY_PARAMS : 1,
                     nb_colors = dim_size[(uint32_t)SkyDataShapeIdx::WAVELENGTH];

        size_t dst_idx = 0;
        double* buffer = (double*)calloc(tensor_size, sizeof(double));

        // Converts from (11 x 2 x 10 x 6 x ...) to (10 x 2 x 6 x 11 x ...)
        for (size_t t = 0; t < NB_TURBIDITY; ++t) {

            for (size_t a = 0; a < NB_ALBEDO; ++a) {

                for (size_t ctrl_idx = 0; ctrl_idx < NB_SKY_CTRL_PTS; ++ctrl_idx) {

                    for (size_t color_idx = 0; color_idx < nb_colors; ++color_idx) {

                        for (size_t param_idx = 0; param_idx < nb_params; ++param_idx) {

                            size_t src_global_offset = a * (NB_TURBIDITY * NB_SKY_CTRL_PTS * nb_params) +
                                                       t * (NB_SKY_CTRL_PTS * nb_params) +
                                                       ctrl_idx * nb_params +
                                                       param_idx;
                            buffer[dst_idx] = p_dataset[color_idx][src_global_offset];
                            ++dst_idx;
                        }
                    }
                }
            }
        }

        // Write the data from the dataset
        file.write_array(buffer, tensor_size);

        free(buffer);
        file.close();
    }

    /**
     * Generates the datasets files from the original, this function should not
     * be called for each render job, only when the dataset files are lost.
     *
     * @param path where the dataset files will be written
     *             The path should include the prefix of the filename
     */
    void write_sun_sky_model_data(const std::string &path) {
        write_sky_data(path + "sky_spec_params.bin", f_spectral);
        write_sky_data(path + "sky_spec_rad.bin", l_spectral);
        write_sky_data(path + "sky_rgb_params.bin", f_RGB);
        write_sky_data(path + "sky_rgb_rad.bin", l_RGB);
        write_sky_data(path + "sky_xyz_params.bin", f_XYZ);
        write_sky_data(path + "sky_xyz_rad.bin", l_XYZ);
        write_sun_data_spectral(path + "sun_spec_rad.bin");
        write_sun_data_rgb(path + "sun_rgb_rad.bin");
        write_limb_darkening_data(path + "sun_spec_ld.bin");
    }


    // Here is the python script used to generate the truncated gaussian dataset file
    /*
    import sys; sys.path.insert(0, "build/python")

    import numpy as np
    import pandas as pd

    import mitsuba as mi

    mi.set_variant("llvm_rgb")

    filename = "<path to>/model_hosek.csv"
    destination_folder = ...

    # Delete unused data
    df = pd.read_csv(filename)
    df.pop('RMSE')
    df.pop('MAE')
    df.pop('Volume')
    df.pop('Normalization')
    df.pop('Azimuth')

    arr = df.to_numpy()

    # Sort the data by turbidity and elevation
    sort_args = np.lexsort([arr[::, 1], arr[::, 0]])
    simplified_arr = arr[sort_args, 2:]

    # Convert the elevation to zenith angle on mu_theta
    simplified_arr[::, 1] = np.pi/2 - simplified_arr[::, 1]

    shape = (9, 30, 5, 5)
    mi.array_to_file(f"{destination_folder}/tgmm_tables.bin", mi.Float(np.ravel(simplified_arr)), shape)
    */

NAMESPACE_END(mitsuba)