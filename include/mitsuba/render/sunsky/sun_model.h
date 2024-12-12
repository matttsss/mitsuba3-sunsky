#pragma once

#include <vector>

#include <mitsuba/core/fstream.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/distr_1d.h>

#include <drjit/sphere.h>

#define SUN_APP_RADIUS 0.5358
#define EARTH_MEAN_RADIUS 6371.01   // In km
#define ASTRONOMICAL_UNIT 149597890 // In km

NAMESPACE_BEGIN(mitsuba)

    // k_o Spectrum table from pg 127, MI.
    float k_oWavelengths[64] = {
        300, 305, 310, 315, 320, 325, 330, 335, 340, 345,
        350, 355, 445, 450, 455, 460, 465, 470, 475, 480,
        485, 490, 495, 500, 505, 510, 515, 520, 525, 530,
        535, 540, 545, 550, 555, 560, 565, 570, 575, 580,
        585, 590, 595, 600, 605, 610, 620, 630, 640, 650,
        660, 670, 680, 690, 700, 710, 720, 730, 740, 750,
        760, 770, 780, 790
    };

    float k_oAmplitudes[65] = {
        10.0, 4.8, 2.7, 1.35, .8, .380, .160, .075, .04, .019, .007,
        .0, .003, .003, .004, .006, .008, .009, .012, .014, .017,
        .021, .025, .03, .035, .04, .045, .048, .057, .063, .07,
        .075, .08, .085, .095, .103, .110, .12, .122, .12, .118,
        .115, .12, .125, .130, .12, .105, .09, .079, .067, .057,
        .048, .036, .028, .023, .018, .014, .011, .010, .009,
        .007, .004, .0, .0
    };

    // k_g Spectrum table from pg 130, MI.
    float k_gWavelengths[4] = {
        759, 760, 770, 771
    };

    float k_gAmplitudes[4] = {
        0, 3.0, 0.210, 0
    };

    // k_wa Spectrum table from pg 130, MI.
    float k_waWavelengths[13] = {
        689, 690, 700, 710, 720,
        730, 740, 750, 760, 770,
        780, 790, 800
    };

    float k_waAmplitudes[13] = {
        0, 0.160e-1, 0.240e-1, 0.125e-1,
        0.100e+1, 0.870, 0.610e-1, 0.100e-2,
        0.100e-4, 0.100e-4, 0.600e-3,
        0.175e-1, 0.360e-1
    };

    /* Wavelengths corresponding to the table below */
    float solWavelengths[38] = {
        380, 390, 400, 410, 420, 430, 440, 450,
        460, 470, 480, 490, 500, 510, 520, 530,
        540, 550, 560, 570, 580, 590, 600, 610,
        620, 630, 640, 650, 660, 670, 680, 690,
        700, 710, 720, 730, 740, 750
    };

    /* Solar amplitude in watts / (m^2 * nm * sr) */
    float solAmplitudes[38] = {
        16559.0, 16233.7, 21127.5, 25888.2, 25829.1,
        24232.3, 26760.5, 29658.3, 30545.4, 30057.5,
        30663.7, 28830.4, 28712.1, 27825.0, 27100.6,
        27233.6, 26361.3, 25503.8, 25060.2, 25311.6,
        25355.9, 25134.2, 24631.5, 24173.2, 23685.3,
        23212.1, 22827.7, 22339.8, 21970.2, 21526.7,
        21097.9, 20728.3, 20240.4, 19870.8, 19427.2,
        19072.4, 18628.9, 18259.2
    };

    float solarWavelenghts[91] = {
        350, 355, 360, 365, 370, 375, 380, 385,
        390, 395, 400, 405, 410, 415, 420, 425,
        430, 435, 440, 445, 450, 455, 460, 465,
        470, 475, 480, 485, 490, 495, 500, 505,
        510, 515, 520, 525, 530, 535, 540, 545,
        550, 555, 560, 565, 570, 575, 580, 585,
        590, 595, 600, 605, 610, 615, 620, 625,
        630, 635, 640, 645, 650, 655, 660, 665,
        670, 675, 680, 685, 690, 695, 700, 705,
        710, 715, 720, 725, 730, 735, 740, 745,
        750, 755, 760, 765, 770, 775, 780, 785,
        790, 795, 800
    };

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
