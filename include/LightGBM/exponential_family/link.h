

#ifndef LIGHTGBM_EXPONENTIAL_FAMILY_LINK_
#define LIGHTGBM_EXPONENTIAL_FAMILY_LINK_

#include <boost/math/distributions/normal.hpp>

#define _USE_MATH_DEFINES

namespace LightGBM {

class ExponentialFamilyLink {
 public:
  /*! \brief virtual constructor */
  ExponentialFamilyLink() {}
  /*! \brief virtual destructor */
  virtual ~ExponentialFamilyLink() {}
  virtual const char* GetName() const = 0;
  virtual double Function(const double&) = 0;
  virtual double Inverse(const double&) = 0;
  virtual double InverseFirstDerivative(const double&) = 0;
  virtual double InverseSecondDerivative(const double&) = 0;
};

class Logit : public ExponentialFamilyLink {
public:
    Logit() {}
    ~Logit() {}
    const char* GetName() const override {
        return "logit";
    }
    inline double Function(const double& prob) override {
        return std::log(prob / (1 - prob));
    }

    inline double Inverse(const double& score) override {
        return 1.0f / (1.0f + std::exp(-score));
    }

    inline double InverseFirstDerivative(const double& score) override {
        const double denominator = 1 + std::exp(-score);
        return std::exp(-score) / (denominator * denominator);
    }

    inline double InverseSecondDerivative(const double& score) override {
        const double denominator = 1 + std::exp(-score);
        return 2*std::exp(-2*score) / (denominator * denominator * denominator) - std::exp(-score) / (denominator * denominator);
    }
};

class Probit : public ExponentialFamilyLink {
public:
    Probit() {}
    ~Probit() {}
    const char* GetName() const override {
        return "probit";
    }
    inline double Function(const double& prob) override {
        return boost::math::quantile(boost::math::normal(0.0, 1.0), prob);
    }

    inline double Inverse(const double& score) override {
        return 0.5 * erfc(-score * M_SQRT1_2);
    }

    inline double InverseFirstDerivative(const double& score) override {
        return 1.0f / std::sqrt(2 * M_PI) * std::exp(-0.5f * score * score);
    }

    inline double InverseSecondDerivative(const double& score) override {
        return -score * 1.0f / std::sqrt(2 * M_PI) * std::exp(-0.5f * score * score);
    }
};

class Cauchit : public ExponentialFamilyLink {
public:
    Cauchit() {}
    ~Cauchit() {}
    const char* GetName() const override {
        return "cauchit";
    }
    inline double Function(const double& prob) override {
        return tan(M_PI * (prob - 0.5f));
    }

    inline double Inverse(const double& score) override {
        return 0.5f + M_1_PI * atan(score);
    }

    inline double InverseFirstDerivative(const double& score) override {
        return M_1_PI / (1 + score * score);
    }

    inline double InverseSecondDerivative(const double& score) override {
        const double one_plus_score_squared = (1 + score * score);
        return -2 * score * M_1_PI / (one_plus_score_squared * one_plus_score_squared);
    }
};

class CLogLog : public ExponentialFamilyLink {
public:
    CLogLog() {}
    ~CLogLog() {}
    const char* GetName() const override {
        return "cloglog";
    }
    inline double Function(const double& prob) override {
        return std::log(-std::log(1.0f - prob));
    }

    inline double Inverse(const double& score) override {
        return 1.0f - std::exp(-std::exp(score));
    }

    inline double InverseFirstDerivative(const double& score) override {
        return std::exp(score - std::exp(score));
    }

    inline double InverseSecondDerivative(const double& score) override {
        return std::exp(score - std::exp(score)) * (1 - std::exp(score));
    }
};

class LogLog : public ExponentialFamilyLink {
public:
    LogLog() {}
    ~LogLog() {}
    const char* GetName() const override {
        return "loglog";
    }
    inline double Function(const double& prob) override {
        return -std::log(-std::log(prob));
    }

    inline double Inverse(const double& score) override {
        return std::exp(-std::exp(-score));
    }

    inline double InverseFirstDerivative(const double& score) override {
        return std::exp(-score - std::exp(-score));;
    }

    inline double InverseSecondDerivative(const double& score) override {
        return std::exp(-score - std::exp(-score)) * (std::exp(-score) - 1);
    }
};

}  // namespace LightGBM

#endif   // LIGHTGBM_EXPONENTIAL_FAMILY_LINK_