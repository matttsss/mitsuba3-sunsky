#pragma once

#include <utility>
#include <vector>

#include <mitsuba/core/fstream.h>
#include <mitsuba/core/logger.h>

#include <drjit/dynamic.h>
#include <drjit/tensor.h>
#include <drjit/sphere.h>

#include "skymodeldata.h"

NAMESPACE_BEGIN(mitsuba)

    #define NB_TURBIDITY 10
    #define NB_CTRL_PT   6
    #define NB_PARAMS    9
    #define NB_ALBEDO    2

    // ================================================================================================
    // ====================================== Sun Coordinates =========================================
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

    #define EARTH_MEAN_RADIUS 6371.01   // In km
    #define ASTRONOMICAL_UNIT 149597890 // In km

    /**
     * \brief Compute the elevation and azimuth of the sun as seen by an observer
     * at \c location at the date and time specified in \c dateTime.
     *
     * Based on "Computing the Solar Vector" by Manuel Blanco-Muriel,
     * Diego C. Alarcon-Padilla, Teodoro Lopez-Moratalla, and Martin Lara-Coira,
     * in "Solar energy", vol 27, number 5, 2001 by Pergamon Press.
     */
    template <typename Float>
    Vector<dr::value_t<Float>, 3> compute_sun_coordinates(const DateTimeRecord<Float> &dateTime, const LocationRecord<Float> &location) {
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

        return from_spherical(Point<dr::value_t<Float>, 2>(azimuth, elevation));
    }

    template <uint32_t arraySize, typename ScalarFloat>
    static ScalarFloat interpolate_wavelength(const double* amplitudes, const double* wavelengths, ScalarFloat wavelength) {
        static_assert(arraySize >= 2);

        uint32_t idx = (uint32_t) -1;
        for (uint32_t i = 0; i < arraySize - 1; ++i) {
            if (wavelengths[i] <= wavelength && wavelength <= wavelengths[i + 1]) {
                idx = i;
                break;
            }
        }

        if (idx == (uint32_t) -1)
            return 0.0f;

        const ScalarFloat t = (wavelength - wavelengths[idx]) / (wavelengths[idx + 1] - wavelengths[idx]);
        return dr::lerp(amplitudes[idx], amplitudes[idx + 1], t);
    }

    template <typename ScalarFloat>
    auto compute_sun_radiance(ScalarFloat sun_theta, ScalarFloat turbidity) {
        std::vector<ScalarFloat> sun_radiance(91);
        ScalarFloat beta = 0.04608365822050f * turbidity - 0.04586025928522f;

        // Relative Optical Mass
        ScalarFloat m = 1.0f / (dr::cos(sun_theta) + 0.15f * dr::pow(93.885f - dr::InvPi<ScalarFloat> * sun_theta * 180.0f, -1.253f));

        size_t i;
        ScalarFloat lbda;
        for (i = 0, lbda = 350; i < 91; i++, lbda += 5) {
            // Rayleigh Scattering
            // Results agree with the graph (pg 115, MI) */
            ScalarFloat tauR = dr::exp(-m * 0.008735f * dr::pow(lbda/1000.0f, -4.08f));

            // Aerosol (water + dust) attenuation
            // beta - amount of aerosols present
            // alpha - ratio of small to large particle sizes. (0:4,usually 1.3)
            // Results agree with the graph (pg 121, MI)
            const ScalarFloat alpha = 1.3f;
            ScalarFloat tauA = dr::exp(-m * beta * dr::pow(lbda/1000.0f, -alpha));  // lambda should be in um

            // Attenuation due to ozone absorption
            // lOzone - amount of ozone in cm(NTP)
            // Results agree with the graph (pg 128, MI)
            const ScalarFloat lOzone = .35f,
                        k_oValue = interpolate_wavelength<64>(k_oAmplitudes, k_oWavelengths, lbda);

            ScalarFloat tauO = dr::exp(-m * k_oValue * lOzone);

            // Attenuation due to mixed gases absorption
            // Results agree with the graph (pg 131, MI)
            const ScalarFloat k_gValue = interpolate_wavelength<4>(k_gAmplitudes, k_gWavelengths, lbda);
            ScalarFloat tauG = dr::exp(-1.41f * k_gValue * m / dr::pow(1 + 118.93f * k_gValue * m, 0.45f));

            // Attenuation due to water vapor absorbtion
            // w - precipitable water vapor in centimeters (standard = 2)
            // Results agree with the graph (pg 132, MI)
            const ScalarFloat w = 2.0f,
                        k_waValue = interpolate_wavelength<13>(k_waAmplitudes, k_waWavelengths, lbda);
            ScalarFloat tauWA = dr::exp(-0.2385f * k_waValue * w * m / dr::pow(1 + 20.07f * k_waValue * w * m, 0.45f));

            ScalarFloat solValue = interpolate_wavelength<38>(solAmplitudes, solWavelengths, lbda);
            sun_radiance[i] = dr::maximum(solValue * tauR * tauA * tauO * tauG * tauWA, 0.0f);
        }


        return sun_radiance;
    }

    // ================================================================================================
    // =========================================== FILE I/O ===========================================
    // ================================================================================================

    #define F_DIM 5
    #define L_DIM 4

    enum DataSetShapeIdx { WAVELENGTH = 0, ALBEDO, TURBIDITY, CTRL_PT, PARAMS };

    constexpr size_t f_spec_shape[F_DIM] = {11, NB_ALBEDO, NB_TURBIDITY, NB_CTRL_PT, NB_PARAMS};
    constexpr size_t l_spec_shape[L_DIM] = {11, NB_ALBEDO, NB_TURBIDITY, NB_CTRL_PT};

    constexpr size_t f_tri_shape[F_DIM] = {3, NB_ALBEDO, NB_TURBIDITY, NB_CTRL_PT, NB_PARAMS};
    constexpr size_t l_tri_shape[L_DIM] = {3, NB_ALBEDO, NB_TURBIDITY, NB_CTRL_PT};

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


    void write_tensor_data_v2(const std::string &path, const Dataset& dataset) {
        const auto [nb_dims, dim_size, p_dataset] = dataset;
        FileStream file(path, FileStream::EMode::ETruncReadWrite);

        // Write headers
        file.write("SKY", 3);
        file.write((uint32_t)2);

        // Write tensor dimensions
        file.write(nb_dims);

        size_t tensor_size = 1;
        for (size_t dim = 0; dim < nb_dims; ++dim)
            tensor_size *= dim_size[dim];


        // Write reordered shapes
        file.write(dim_size[TURBIDITY]);
        file.write(dim_size[ALBEDO]);
        file.write(dim_size[CTRL_PT]);
        file.write(dim_size[WAVELENGTH]);

        if (nb_dims == F_DIM)
            file.write(dim_size[PARAMS]);


        const size_t nb_params = nb_dims == F_DIM ? NB_PARAMS : 1,
                     nb_colors = dim_size[WAVELENGTH];

        double* buffer = (double*)calloc(tensor_size, sizeof(double));

        // Converts from (11 x 2 x 10 x 6 x ...) to (10 x 2 x 6 x 11 x ...)
        for (size_t t = 0; t < NB_TURBIDITY; ++t) {

            for (size_t a = 0; a < NB_ALBEDO; ++a) {

                for (size_t ctrl_idx = 0; ctrl_idx < NB_CTRL_PT; ++ctrl_idx) {

                    for (size_t color_idx = 0; color_idx < nb_colors; ++color_idx) {

                        for (size_t param_idx = 0; param_idx < nb_params; ++param_idx) {
                            size_t dest_global_offset = t * (NB_ALBEDO * NB_CTRL_PT * nb_colors * nb_params) +
                                                        a * (NB_CTRL_PT * nb_colors * nb_params) +
                                                        ctrl_idx * nb_colors * nb_params +
                                                        color_idx * nb_params +
                                                        param_idx;
                            size_t src_global_offset = a * (NB_TURBIDITY * NB_CTRL_PT * nb_params) +
                                                       t * (NB_CTRL_PT * nb_params) +
                                                       ctrl_idx * nb_params +
                                                       param_idx;
                            buffer[dest_global_offset] = p_dataset[color_idx][src_global_offset];

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

    void write_sky_model_data_v2(const std::string &path) {
        write_tensor_data_v2(path + "_v2_spec.bin", f_spectral);
        write_tensor_data_v2(path + "_v2_spec_rad.bin", l_spectral);
        write_tensor_data_v2(path + "_v2_rgb.bin", f_RGB);
        write_tensor_data_v2(path + "_v2_rgb_rad.bin", l_RGB);
    }


    void write_tensor_data_v1(const std::string &path, const Dataset& dataset) {
        const auto [nb_dims, dim_size, p_dataset] = dataset;
        FileStream file(path, FileStream::EMode::ETruncReadWrite);

        // Write headers
        file.write("SKY", 3);
        file.write((uint32_t)1);

        // Write tensor dimensions
        file.write(nb_dims);

        size_t tensor_size = 1;
        for (size_t dim = 0; dim < nb_dims; ++dim)
            tensor_size *= dim_size[dim];


        // Write reordered shapes
        file.write(dim_size[ALBEDO]);
        file.write(dim_size[TURBIDITY]);
        file.write(dim_size[CTRL_PT]);
        file.write(dim_size[WAVELENGTH]);

        if (nb_dims == F_DIM)
            file.write(dim_size[PARAMS]);


        const size_t nb_params = nb_dims == F_DIM ? NB_PARAMS : 1,
                     nb_colors = dim_size[WAVELENGTH];

        double* buffer = (double*)calloc(tensor_size, sizeof(double));

        // Converts from (11 x 2 x 10 x 6 x ...) to (2 x 10 x 6 x 11 x ...)
        for (size_t a = 0; a < NB_ALBEDO; ++a) {

            for (size_t t = 0; t < NB_TURBIDITY; ++t) {

                for (size_t ctrl_idx = 0; ctrl_idx < NB_CTRL_PT; ++ctrl_idx) {

                    for (size_t color_idx = 0; color_idx < nb_colors; ++color_idx) {

                        for (size_t param_idx = 0; param_idx < nb_params; ++param_idx) {
                            size_t dest_global_offset = a * (NB_TURBIDITY * NB_CTRL_PT * nb_colors * nb_params) +
                                                        t * (NB_CTRL_PT * nb_colors * nb_params) +
                                                        ctrl_idx * nb_colors * nb_params +
                                                        color_idx * nb_params +
                                                        param_idx;
                            size_t src_global_offset = a * (NB_TURBIDITY * NB_CTRL_PT * nb_params) +
                                                       t * (NB_CTRL_PT * nb_params) +
                                                       ctrl_idx * nb_params +
                                                       param_idx;
                            buffer[dest_global_offset] = p_dataset[color_idx][src_global_offset];

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

    void write_sky_model_data_v1(const std::string &path) {
        write_tensor_data_v1(path + "_v1_spec.bin", f_spectral);
        write_tensor_data_v1(path + "_v1_spec_rad.bin", l_spectral);
        write_tensor_data_v1(path + "_v1_rgb.bin", f_RGB);
        write_tensor_data_v1(path + "_v1_rgb_rad.bin", l_RGB);
    }

    void write_tensor_data_v0(const std::string &path, const Dataset& dataset) {
        const auto [nb_dims, dim_size, p_dataset] = dataset;
        FileStream file(path, FileStream::EMode::ETruncReadWrite);

        // Write headers
        file.write("SKY", 3);
        file.write((uint32_t)0);

        // Write tensor dimensions
        file.write(nb_dims);
        size_t lane_size = 1;
        for (size_t dim = 0; dim < nb_dims; ++dim) {
            file.write(dim_size[dim]);
            lane_size *= dim ? dim_size[dim] : 1;
        }

        // Write the data from the dataset
        for (size_t lane_idx = 0; lane_idx < dim_size[0]; ++lane_idx)
            file.write_array(p_dataset[lane_idx], lane_size);

        file.close();
    }

    void write_sky_model_data_v0(const std::string &path) {
        write_tensor_data_v0(path + "_v0_spec.bin", f_spectral);
        write_tensor_data_v0(path + "_v0_spec_rad.bin", l_spectral);
        write_tensor_data_v0(path + "_v0_rgb.bin", f_RGB);
        write_tensor_data_v0(path + "_v0_rgb_rad.bin", l_RGB);
    }

    template<typename Float>
    auto array_from_file_d(const std::string &path) {
        using FloatStorage  = dr::DynamicArray<Float>;
        using DoubleStorage = dr::float64_array_t<FloatStorage>;

        FileStream file(path, FileStream::EMode::ERead);

        // =============== Read headers ===============
        char text_buff[5] = "";
        file.read(text_buff, 3);
        if (strcmp(text_buff, "SKY"))
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
        double* buffer = static_cast<double *>(
            calloc(nb_elements, sizeof(double)));

        file.read_array(buffer, nb_elements);

        DoubleStorage data_d = dr::load<DoubleStorage>(buffer, nb_elements);
        FloatStorage data_v = FloatStorage(data_d);

        file.close();
        free(buffer);

        return std::pair<std::vector<size_t>, FloatStorage>(shape, data_v);
    }

    template<typename FileType, typename OutType>
    std::vector<OutType> array_from_file(const std::string &path) {
        FileStream file(path, FileStream::EMode::ERead);

        // =============== Read headers ===============
        char text_buff[5] = "";
        file.read(text_buff, 3);
        if (strcmp(text_buff, "SKY"))
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
        FileType* buffer = static_cast<FileType *>(
            calloc(nb_elements, sizeof(FileType)));

        file.read_array(buffer, nb_elements);
        std::vector<FileType> data_f(buffer, buffer + nb_elements);

        file.close();
        free(buffer);

        return std::vector<OutType>(data_f.begin(), data_f.end());
    }

    template<typename Float>
    auto array_from_file_f(const std::string &path) {
        using FloatStorage  = DynamicBuffer<Float>;

        FileStream file(path, FileStream::EMode::ERead);

        // =============== Read headers ===============
        char text_buff[5] = "";
        file.read(text_buff, 3);
        if (strcmp(text_buff, "SKY"))
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
        float* buffer = static_cast<float *>(
            calloc(nb_elements, sizeof(float)));

        file.read_array(buffer, nb_elements);

        FloatStorage data_v = dr::load<FloatStorage>(buffer, nb_elements);

        file.close();
        free(buffer);

        return std::pair<std::vector<size_t>, FloatStorage>(shape, data_v);
    }

    template<typename Float>
    void array_to_file(const std::string &path, const DynamicBuffer<Float>& data, const std::vector<size_t>& shape = {}) {
        std::vector<size_t> _shape = shape.empty() ? std::vector<size_t>{data.size()} : shape;
        if (_shape.empty())
            _shape.push_back(data.size());

        FileStream file(path, FileStream::EMode::ETruncReadWrite);

        // =============== Write headers ===============
        // Write headers
        file.write("SKY", 3);
        file.write((uint32_t) 0);

        // =============== Write dimensions ===============
        file.write((size_t) _shape.size());

        for (size_t dim_length : _shape) {
            file.write(dim_length);

            if (!dim_length)
                Throw("Got dimension with 0 elements");
        }

        // ==================== Write data ====================
        file.write_array(data.data(), data.size());

        file.close();
    }


    template<typename Float>
    auto tensor_from_file(const std::string &path) {
        using FloatStorage  = DynamicBuffer<Float>;
        using DoubleStorage = dr::float64_array_t<FloatStorage>;
        using TensorXf      = dr::Tensor<FloatStorage>;

        FileStream file(path, FileStream::EMode::ERead);

        // =============== Read headers ===============
        char text_buff[5] = "";
        file.read(text_buff, 3);
        if (strcmp(text_buff, "SKY"))
            Throw("OUPSSS wrong file");

        // Read version
        uint32_t version;
        file.read(version);

        // =============== Read tensor dimensions ===============
        size_t nb_dims = 0;
        file.read(nb_dims);

        size_t nb_elements = 1;
        size_t shape[nb_dims];
        for (size_t dim = 0; dim < nb_dims; ++dim) {
            file.read(shape[dim]);

            if (!shape[dim])
                Throw("Got dimension with 0 elements");

            nb_elements *= shape[dim];
        }

        // ==================== Read data ====================
        double* buffer = static_cast<double *>(
            calloc(nb_elements, sizeof(double)));

        file.read_array(buffer, nb_elements);

        DoubleStorage data_d = dr::load<DoubleStorage>(buffer, nb_elements);
        FloatStorage data_v = FloatStorage(data_d);

        file.close();
        free(buffer);

        return TensorXf(data_v, nb_dims, shape);
    }


    // ================================================================================================
    // ======================================= HELPER FUNCTIONS =======================================
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

    template<typename ScalarFloat>
    std::vector<ScalarFloat> lerp_vectors(const std::vector<ScalarFloat>& a, const std::vector<ScalarFloat>& b, const ScalarFloat& t) {
        assert(a.size() == b.size());

        std::vector<ScalarFloat> res(a.size());
        for (size_t i = 0; i < a.size(); ++i)
            res[i] = dr::lerp(a[i], b[i], t);

        return res;
    }

NAMESPACE_END(mitsuba)
