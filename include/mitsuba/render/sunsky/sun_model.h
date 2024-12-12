#pragma once

#include <vector>

#include <mitsuba/core/fstream.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/distr_1d.h>

#include <drjit/sphere.h>

#include "sky_data.h"

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

    template <typename Wavelength>
    struct SunParameters {
        using SunParamSpectrum = IrregularContinuousDistribution<Wavelength>;

        SunParamSpectrum k_o;
        SunParamSpectrum k_g;
        SunParamSpectrum k_wa;
        SunParamSpectrum sol;

        SunParameters() {
            k_o = SunParamSpectrum(k_oWavelengths, k_oAmplitudes, 64);
            k_g = SunParamSpectrum(k_gWavelengths, k_gAmplitudes, 4);
            k_wa = SunParamSpectrum(k_waWavelengths, k_waAmplitudes, 13);
            sol = SunParamSpectrum(solWavelengths, solAmplitudes, 38);
        }

    };

    template <typename ScalarFloat>
    auto compute_sun_radiance(SunParameters<ScalarFloat> sunParams, ScalarFloat sun_theta, ScalarFloat turbidity) {
        // TODO originaly in double
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
            const ScalarFloat lOzone = .35f;

            ScalarFloat tauO = dr::exp(-m * sunParams.k_o.eval_pdf(lbda) * lOzone);

            // Attenuation due to mixed gases absorption
            // Results agree with the graph (pg 131, MI)
            const ScalarFloat k_gValue = sunParams.k_g.eval_pdf(lbda);
            ScalarFloat tauG = dr::exp(-1.41f * k_gValue * m / dr::pow(1 + 118.93f * k_gValue * m, 0.45f));

            // Attenuation due to water vapor absorbtion
            // w - precipitable water vapor in centimeters (standard = 2)
            // Results agree with the graph (pg 132, MI)
            const ScalarFloat w = 2.0f,
                        k_waValue = sunParams.k_wa.eval_pdf(lbda);
            ScalarFloat tauWA = dr::exp(-0.2385f * k_waValue * w * m / dr::pow(1 + 20.07f * k_waValue * w * m, 0.45f));

            ScalarFloat solValue = sunParams.sol.eval_pdf(lbda);
            sun_radiance[i] = dr::maximum(solValue * tauR * tauA * tauO * tauG * tauWA, 0.0f);
        }


        return sun_radiance;
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

NAMESPACE_END(mitsuba)
