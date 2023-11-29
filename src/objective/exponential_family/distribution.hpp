#ifndef LIGHTGBM_EXPONENTIAL_FAMILY_LOG_LIKELIHOOD_
#define LIGHTGBM_EXPONENTIAL_FAMILY_LOG_LIKELIHOOD_
#include "link.hpp"

namespace LightGBM
{

  class ExponentialFamilyDistribution
  {
  public:
    /*! \brief virtual constructor */
    ExponentialFamilyDistribution() = default;
    ExponentialFamilyDistribution(std::shared_ptr<ExponentialFamilyLink> exponential_family_link)
    {
      link_ = exponential_family_link;
    }
    /*! \brief virtual destructor */
    virtual ~ExponentialFamilyDistribution() = default;
    virtual const char *GetName() const = 0;
    virtual double NegativeLogLikelihood(const float &, const double &) = 0;
    virtual double NegativeLogLikelihoodFirstDerivative(const float &, const double &) = 0;
    virtual double NegativeLogLikelihoodSecondDerivative(const float &, const double &) = 0;

  protected:
    std::shared_ptr<ExponentialFamilyLink> link_;
  };

  class Bernoulli : public ExponentialFamilyDistribution
  {
  public:
    Bernoulli() {}
    Bernoulli(std::shared_ptr<ExponentialFamilyLink> exponential_family_link)
    {
      link_ = exponential_family_link;
    }
    /*! \brief Destructor */
    ~Bernoulli() {}
    const char *GetName() const
    {
      return "bernoulli";
    }
    double inline NegativeLogLikelihood(const float &actual, const double &score) override
    {
      if (actual > 0)
      {
        return -std::log(link_->Inverse(score));
      }
      return -std::log1p(-link_->Inverse(score));
    }

    double inline NegativeLogLikelihoodFirstDerivative(const float &actual, const double &score) override
    {
      double const binary = actual > 0 ? 1 : 0;
      return -link_->InverseFirstDerivative(score) / (link_->Inverse(score) - 1 + binary);
    }
    double inline NegativeLogLikelihoodSecondDerivative(const float &actual, const double &score) override
    {
      double const binary = actual > 0 ? 1 : 0;
      double const intermediate = link_->Inverse(score) - 1 + binary;
      double const link_first_derivative = link_->InverseFirstDerivative(score);

      return -(intermediate * link_->InverseSecondDerivative(score) - (link_first_derivative * link_first_derivative)) / (intermediate * intermediate);
    }
  };

  class Guassian : public ExponentialFamilyDistribution
  {
  public:
    Guassian() {}
    Guassian(std::shared_ptr<ExponentialFamilyLink> exponential_family_link)
    {
      link_ = exponential_family_link;
    }
    /*! \brief Destructor */
    ~Guassian() {}
    const char *GetName() const
    {
      return "guassian";
    }
    double inline NegativeLogLikelihood(const float &actual, const double &score) override
    {
      return 0.5f * ((actual - link_->Inverse(score)) * (actual - link_->Inverse(score)) + std::log(2 * M_PI));
    }

    double inline NegativeLogLikelihoodFirstDerivative(const float &actual, const double &score) override
    {
      return link_->InverseFirstDerivative(score) * (link_->Inverse(score) - actual);
    }
    double inline NegativeLogLikelihoodSecondDerivative(const float &actual, const double &score) override
    {
      const double link_first_derivate = link_->InverseFirstDerivative(score);
      return (link_->Inverse(score) - actual) * link_->InverseSecondDerivative(score) + link_first_derivate * link_first_derivate;
    }
  };

} // namespace LightGBM

#endif // LIGHTGBM_EXPONENTIAL_FAMILY_LOG_LIKELIHOOD_