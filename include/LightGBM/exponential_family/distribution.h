#ifndef LIGHTGBM_EXPONENTIAL_FAMILY_LOG_LIKELIHOOD_
#define LIGHTGBM_EXPONENTIAL_FAMILY_LOG_LIKELIHOOD_
#include <LightGBM/exponential_family/link.h>

namespace LightGBM {

class ExponentialFamilyDistribution {
 public:
  /*! \brief virtual constructor */
  ExponentialFamilyDistribution() = default;
  ExponentialFamilyDistribution(std::shared_ptr<ExponentialFamilyLink> exponential_family_link) {
    link_ = exponential_family_link;
  }
  /*! \brief virtual destructor */
  virtual ~ExponentialFamilyDistribution() = default;
  virtual const char* GetName() const = 0;
  virtual double NegativeLogLikelihood(const float&, const double&) = 0;
  virtual double NegativeLogLikelihoodFirstDerivative(const float&, const double&) = 0;
  virtual double NegativeLogLikelihoodSecondDerivative(const float&, const double&) = 0;

  private:
    std::shared_ptr<ExponentialFamilyLink> link_;

  friend class Bernoulli;
};

class Bernoulli : public ExponentialFamilyDistribution {
  public:
  Bernoulli() {}
  Bernoulli(std::shared_ptr<ExponentialFamilyLink> exponential_family_link) {
    link_ = exponential_family_link;
  }
  /*! \brief Destructor */
  ~Bernoulli() {}
  const char* GetName() const {
    return "bernoulli";
  }
  double inline NegativeLogLikelihood(const float& actual, const double& score) override {
    if (actual > 0) {
      return -std::log(link_->Inverse(score));
    }
    return -std::log(1 - link_->Inverse(score));
  }

  double inline NegativeLogLikelihoodFirstDerivative(const float& actual, const double& score) override {
    double const binary = actual > 0 ? 1 : 0;
    return -link_->InverseFirstDerivative(score) / (link_->Inverse(score) - 1 + binary);
  }
  double inline NegativeLogLikelihoodSecondDerivative(const float& actual, const double& score) override {
    double const binary = actual > 0 ? 1 : 0;
    double const intermediate = link_->Inverse(score) - 1 + binary;
    double const link_first_derivative = link_->InverseFirstDerivative(score);

    return -(intermediate * link_->InverseSecondDerivative(score) - (link_first_derivative * link_first_derivative)) / (intermediate * intermediate);
  }
};

}  // namespace LightGBM

#endif   // LIGHTGBM_EXPONENTIAL_FAMILY_LOG_LIKELIHOOD_