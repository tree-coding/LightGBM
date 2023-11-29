

#ifndef LIGHTGBM_EXPONENTIAL_FAMILY_LINK_
#define LIGHTGBM_EXPONENTIAL_FAMILY_LINK_

#include <boost/math/distributions/normal.hpp>

#define _USE_MATH_DEFINES

namespace LightGBM
{

    class ExponentialFamilyLink
    {
    public:
        /*! \brief virtual constructor */
        ExponentialFamilyLink() {}
        /*! \brief virtual destructor */
        virtual ~ExponentialFamilyLink() {}
        virtual const char *GetName() const = 0;
        virtual double Function(const double &) = 0;
        virtual double Inverse(const double &) = 0;
        virtual double InverseFirstDerivative(const double &) = 0;
        virtual double InverseSecondDerivative(const double &) = 0;
    };

    class IdentityLink : public ExponentialFamilyLink
    {
    public:
        IdentityLink() {}
        ~IdentityLink() {}
        const char *GetName() const override
        {
            return "identity";
        }
        inline double Function(const double &prediction) override
        {
            return prediction;
        }

        inline double Inverse(const double &score) override
        {
            return score;
        }

        inline double InverseFirstDerivative(const double &) override
        {
            return 1.0f;
        }

        inline double InverseSecondDerivative(const double &) override
        {
            return 0.0f;
        }
    };

    class LogLink : public ExponentialFamilyLink
    {
    public:
        LogLink() {}
        ~LogLink() {}
        const char *GetName() const override
        {
            return "log";
        }
        inline double Function(const double &prediction) override
        {
            return std::log(prediction);
        }

        inline double Inverse(const double &score) override
        {
            return std::exp(score);
        }

        inline double InverseFirstDerivative(const double &score) override
        {
            return std::exp(score);
        }

        inline double InverseSecondDerivative(const double &score) override
        {
            return std::exp(score);
        }
    };

    class Log1pLink : public ExponentialFamilyLink
    {
    public:
        Log1pLink() {}
        ~Log1pLink() {}
        const char *GetName() const override
        {
            return "log1p";
        }
        inline double Function(const double &prediction) override
        {
            return Common::Sign(prediction) * std::log1p(std::fabs(prediction));
        }

        inline double Inverse(const double &score) override
        {
            return Common::Sign(score) * std::expm1(std::fabs(score));
        }

        inline double InverseFirstDerivative(const double &score) override
        {
            return std::exp(std::fabs(score));
        }

        inline double InverseSecondDerivative(const double &score) override
        {
            return std::exp(std::fabs(score));
        }
    };

    class SqrtLink : public ExponentialFamilyLink
    {
    public:
        SqrtLink() {}
        ~SqrtLink() {}
        const char *GetName() const override
        {
            return "sqrt";
        }
        inline double Function(const double &prediction) override
        {
            return Common::Sign(prediction) * std::sqrt(std::fabs(prediction));
        }

        inline double Inverse(const double &score) override
        {
            return Common::Sign(score) * score * score;
        }

        inline double InverseFirstDerivative(const double &score) override
        {
            return 2 * std::fabs(score);
        }

        inline double InverseSecondDerivative(const double &score) override
        {
            return 2 * Common::Sign(score);
        }
    };

    class InverseLink : public ExponentialFamilyLink
    {
    public:
        InverseLink() {}
        ~InverseLink() {}
        const char *GetName() const override
        {
            return "inverse";
        }
        inline double Function(const double &prediction) override
        {
            return 1.0f / prediction;
        }

        inline double Inverse(const double &score) override
        {
            return 1.0f / score;
        }

        inline double InverseFirstDerivative(const double &score) override
        {
            return -1.0f / (score * score);
        }

        inline double InverseSecondDerivative(const double &score) override
        {
            return 2.0f / (score * score * score);
        }
    };

    class InverseSquareLink : public ExponentialFamilyLink
    {
    public:
        InverseSquareLink() {}
        ~InverseSquareLink() {}
        const char *GetName() const override
        {
            return "inversesquare";
        }
        inline double Function(const double &prediction) override
        {
            return 1.0f / (prediction * prediction);
        }

        inline double Inverse(const double &score) override
        {
            return 1.0f / std::sqrt(score);
        }

        inline double InverseFirstDerivative(const double &score) override
        {
            return -0.5f / (score * std::sqrt(score));
        }

        inline double InverseSecondDerivative(const double &score) override
        {
            return 0.75f / (score * score * std::sqrt(score));
        }
    };

    class PowerLink : public ExponentialFamilyLink
    {
    public:
        PowerLink() {}
        PowerLink(const double &power) : power_(power), inverse_power_(power == 0 ? 0.0f : 1.0f / power) {}
        ~PowerLink() {}
        const char *GetName() const override
        {
            return "power";
        }
        inline double Function(const double &prediction) override
        {
            return std::pow(prediction, power_);
        }

        inline double Inverse(const double &score) override
        {
            return std::pow(score, inverse_power_);
        }

        inline double InverseFirstDerivative(const double &score) override
        {
            return inverse_power_ * std::pow(score, inverse_power_ - 1);
            ;
        }

        inline double InverseSecondDerivative(const double &score) override
        {
            return inverse_power_ * (inverse_power_ - 1) * std::pow(score, inverse_power_ - 2);
        }

    private:
        const double power_ = 1.0;
        const double inverse_power_ = 1.0;
    };

    class PowerAbsLink : public ExponentialFamilyLink
    {
    public:
        PowerAbsLink() {}
        PowerAbsLink(const double &power) : power_(power), inverse_power_(power == 0 ? 0.0f : 1.0f / power) {}
        ~PowerAbsLink() {}
        const char *GetName() const override
        {
            return "power";
        }
        inline double Function(const double &prediction) override
        {
            return Common::Sign(prediction) * std::pow(std::fabs(prediction), power_);
        }

        inline double Inverse(const double &score) override
        {
            return Common::Sign(score) * std::pow(std::fabs(score), inverse_power_);
        }

        inline double InverseFirstDerivative(const double &score) override
        {
            return inverse_power_ * std::pow(std::fabs(score), inverse_power_ - 1);
            ;
        }

        inline double InverseSecondDerivative(const double &score) override
        {
            return Common::Sign(score) * inverse_power_ * (inverse_power_ - 1) * std::pow(std::fabs(score), inverse_power_ - 2);
        }

    private:
        const double power_ = 1.0;
        const double inverse_power_ = 1.0;
    };

    class LogitLink : public ExponentialFamilyLink
    {
    public:
        LogitLink() {}
        ~LogitLink() {}
        const char *GetName() const override
        {
            return "logit";
        }
        inline double Function(const double &prob) override
        {
            return std::log(prob / (1.0f - prob));
        }

        inline double Inverse(const double &score) override
        {
            return 1.0f / (1.0f + std::exp(-score));
        }

        inline double InverseFirstDerivative(const double &score) override
        {
            const double expit = 1.0f / (1.0f + std::exp(-score));
            return expit * (1.0f - expit);
        }

        inline double InverseSecondDerivative(const double &score) override
        {
            const double expit = 1.0f / (1.0f + std::exp(-score));
            return expit * (1.0f - expit) * (1.0f - 2.0f * expit);
        }
    };

    class ProbitLink : public ExponentialFamilyLink
    {
    public:
        ProbitLink() {}
        ~ProbitLink() {}
        const char *GetName() const override
        {
            return "probit";
        }
        inline double Function(const double &prob) override
        {
            return boost::math::quantile(boost::math::normal(0.0, 1.0), prob);
        }

        inline double Inverse(const double &score) override
        {
            return 0.5 * erfc(-score * M_SQRT1_2);
        }

        inline double InverseFirstDerivative(const double &score) override
        {
            return 1.0f / std::sqrt(2 * M_PI) * std::exp(-0.5f * score * score);
        }

        inline double InverseSecondDerivative(const double &score) override
        {
            return -score * 1.0f / std::sqrt(2 * M_PI) * std::exp(-0.5f * score * score);
        }
    };

    class CauchitLink : public ExponentialFamilyLink
    {
    public:
        CauchitLink() {}
        ~CauchitLink() {}
        const char *GetName() const override
        {
            return "cauchit";
        }
        inline double Function(const double &prob) override
        {
            return tan(M_PI * (prob - 0.5f));
        }

        inline double Inverse(const double &score) override
        {
            return 0.5f + M_1_PI * atan(score);
        }

        inline double InverseFirstDerivative(const double &score) override
        {
            return M_1_PI / (1 + score * score);
        }

        inline double InverseSecondDerivative(const double &score) override
        {
            const double one_plus_score_squared = (1 + score * score);
            return -2 * score * M_1_PI / (one_plus_score_squared * one_plus_score_squared);
        }
    };

    class CLogLogLink : public ExponentialFamilyLink
    {
    public:
        CLogLogLink() {}
        ~CLogLogLink() {}
        const char *GetName() const override
        {
            return "cloglog";
        }
        inline double Function(const double &prob) override
        {
            return std::log(-std::log1p(-prob));
        }

        inline double Inverse(const double &score) override
        {
            return -std::expm1(-std::exp(score));
        }

        inline double InverseFirstDerivative(const double &score) override
        {
            return std::exp(score - std::exp(score));
        }

        inline double InverseSecondDerivative(const double &score) override
        {
            return std::exp(score - std::exp(score)) * -std::expm1(score);
        }
    };

    class LogLogLink : public ExponentialFamilyLink
    {
    public:
        LogLogLink() {}
        ~LogLogLink() {}
        const char *GetName() const override
        {
            return "loglog";
        }
        inline double Function(const double &prob) override
        {
            return -std::log(-std::log(prob));
        }

        inline double Inverse(const double &score) override
        {
            return std::exp(-std::exp(-score));
        }

        inline double InverseFirstDerivative(const double &score) override
        {
            return std::exp(-score - std::exp(-score));
            ;
        }

        inline double InverseSecondDerivative(const double &score) override
        {
            return std::exp(-score - std::exp(-score)) * (std::exp(-score) - 1);
        }
    };

} // namespace LightGBM

#endif // LIGHTGBM_EXPONENTIAL_FAMILY_LINK_