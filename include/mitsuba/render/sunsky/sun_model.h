#pragma once

#include "sunsky_helpers.h"
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/logger.h>

#define SUN_APP_RADIUS 0.5358
#define EARTH_MEAN_RADIUS 6371.01   // In km
#define ASTRONOMICAL_UNIT 149597890 // In km

NAMESPACE_BEGIN(mitsuba)

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


    // ================================================================================================
    // ======================================== Sun Radiance ==========================================
    // ================================================================================================

    template <typename ScalarFloat>
    std::vector<ScalarFloat> compute_sun_params_spectral(const std::vector<ScalarFloat>& sun_radiance_dataset, ScalarFloat turbidity) {
        turbidity = dr::clip(turbidity, 1.f, 10.f);
        uint32_t t_int = dr::floor2int<uint32_t>(turbidity),
                 t_low = dr::maximum(t_int - 1, 0),
                 t_high = dr::minimum(t_low + 1, NB_TURBIDITY - 1);

        uint32_t t_block_size = sun_radiance_dataset.size() / NB_TURBIDITY;

        std::vector<ScalarFloat> res = std::vector<ScalarFloat>(t_block_size);
        for (uint32_t i = 0; i < t_block_size; ++i) {
            ScalarFloat t_low_val  = sun_radiance_dataset[t_low  * t_block_size + i],
                        t_high_val = sun_radiance_dataset[t_high * t_block_size + i];

            res[i] = dr::lerp(t_low_val, t_high_val, turbidity - t_int);
        }

        return res;
    }

    template <typename ScalarFloat>
    std::vector<ScalarFloat> compute_sun_params_rgb(
        const std::vector<ScalarFloat>& sun_radiance_dataset,
        const std::vector<ScalarFloat>& sun_ld_dataset,
        ScalarFloat turbidity) {

        const std::vector<ScalarFloat> sun_radiance_interp =
                    compute_sun_params_spectral(sun_radiance_dataset, turbidity);

        // [COLOR_IDX, SUN_SEGMENT_IDX, SUN_CTRL_PT_IDX, LD_IDX]
        std::vector<ScalarFloat> res = std::vector<ScalarFloat>(3 * NB_SUN_SEGMENTS * NB_SUN_CTRL_PTS * NB_SUN_LD_PARAMS, 0.f);
        for (size_t lambda = 0; lambda < NB_WAVELENGTHS; ++lambda) {
            Color<ScalarFloat, 3> rgb_rectifier = linear_rgb_rec(WAVELENGTHS[lambda]);

            for (size_t rgb_idx = 0; rgb_idx < 3; ++rgb_idx) {
                ScalarFloat rectifier = rgb_rectifier[rgb_idx];

                for (size_t j = 0; j < NB_SUN_LD_PARAMS; ++j) {
                    ScalarFloat limb_darkening_coeff = sun_ld_dataset[lambda * NB_SUN_LD_PARAMS + j];

                    for (size_t i = 0; i < NB_SUN_SEGMENTS; ++i) {

                        for (size_t k = 0; k < NB_SUN_CTRL_PTS; ++k) {

                            const size_t sun_rad_idx = lambda * (NB_SUN_SEGMENTS * NB_SUN_CTRL_PTS) +
                                                       i * NB_SUN_CTRL_PTS +
                                                       k;
                            const size_t dst_idx     = rgb_idx * (NB_SUN_SEGMENTS * NB_SUN_CTRL_PTS * NB_SUN_LD_PARAMS) +
                                                       i * (NB_SUN_CTRL_PTS * NB_SUN_LD_PARAMS) +
                                                       k * NB_SUN_LD_PARAMS +
                                                       j;

                            res[dst_idx] += rectifier * limb_darkening_coeff * sun_radiance_dataset[sun_rad_idx] / NB_WAVELENGTHS;
                        }
                    }
                }
            }
        }

        return res;
    }



NAMESPACE_END(mitsuba)
