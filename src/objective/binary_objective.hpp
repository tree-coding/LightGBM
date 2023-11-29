/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_OBJECTIVE_BINARY_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_BINARY_OBJECTIVE_HPP_

#include <LightGBM/network.h>
#include <LightGBM/objective_function.h>
#include "exponential_family/distribution.hpp"
#include "exponential_family/link.hpp"

#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace LightGBM
{
  /*!
   * \brief Objective function for binary classification
   */
  class BinaryLogloss : public ObjectiveFunction
  {
  public:
    explicit BinaryLogloss(const Config &config,
                           std::function<bool(label_t)> is_pos = nullptr)
        : deterministic_(config.deterministic)
    {
      sigmoid_ = static_cast<double>(config.sigmoid);
      if (sigmoid_ <= 0.0)
      {
        Log::Fatal("Sigmoid parameter %f should be greater than zero", sigmoid_);
      }
      is_unbalance_ = config.is_unbalance;
      scale_pos_weight_ = static_cast<double>(config.scale_pos_weight);
      if (is_unbalance_ && std::fabs(scale_pos_weight_ - 1.0f) > 1e-6)
      {
        Log::Fatal("Cannot set is_unbalance and scale_pos_weight at the same time");
      }
      is_pos_ = is_pos;
      if (is_pos_ == nullptr)
      {
        is_pos_ = [](label_t label)
        { return label > 0; };
      }
    }

    explicit BinaryLogloss(const std::vector<std::string> &strs)
        : deterministic_(false)
    {
      sigmoid_ = -1;
      for (auto str : strs)
      {
        auto tokens = Common::Split(str.c_str(), ':');
        if (tokens.size() == 2)
        {
          if (tokens[0] == std::string("sigmoid"))
          {
            Common::Atof(tokens[1].c_str(), &sigmoid_);
          }
        }
      }
      if (sigmoid_ <= 0.0)
      {
        Log::Fatal("Sigmoid parameter %f should be greater than zero", sigmoid_);
      }
    }

    ~BinaryLogloss() {}

    void Init(const Metadata &metadata, data_size_t num_data) override
    {
      num_data_ = num_data;
      label_ = metadata.label();
      weights_ = metadata.weights();
      data_size_t cnt_positive = 0;
      data_size_t cnt_negative = 0;
// count for positive and negative samples
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+ : cnt_positive, cnt_negative)
      for (data_size_t i = 0; i < num_data_; ++i)
      {
        if (is_pos_(label_[i]))
        {
          ++cnt_positive;
        }
        else
        {
          ++cnt_negative;
        }
      }
      num_pos_data_ = cnt_positive;
      if (Network::num_machines() > 1)
      {
        cnt_positive = Network::GlobalSyncUpBySum(cnt_positive);
        cnt_negative = Network::GlobalSyncUpBySum(cnt_negative);
      }
      need_train_ = true;
      if (cnt_negative == 0 || cnt_positive == 0)
      {
        Log::Warning("Contains only one class");
        // not need to boost.
        need_train_ = false;
      }
      Log::Info("Number of positive: %d, number of negative: %d", cnt_positive, cnt_negative);
      // use -1 for negative class, and 1 for positive class
      label_val_[0] = -1;
      label_val_[1] = 1;
      // weight for label
      label_weights_[0] = 1.0f;
      label_weights_[1] = 1.0f;
      // if using unbalance, change the labels weight
      if (is_unbalance_ && cnt_positive > 0 && cnt_negative > 0)
      {
        if (cnt_positive > cnt_negative)
        {
          label_weights_[1] = 1.0f;
          label_weights_[0] = static_cast<double>(cnt_positive) / cnt_negative;
        }
        else
        {
          label_weights_[1] = static_cast<double>(cnt_negative) / cnt_positive;
          label_weights_[0] = 1.0f;
        }
      }
      label_weights_[1] *= scale_pos_weight_;
    }

    void GetGradients(const double *score, score_t *gradients, score_t *hessians) const override
    {
      if (!need_train_)
      {
        return;
      }
      if (weights_ == nullptr)
      {
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
        for (data_size_t i = 0; i < num_data_; ++i)
        {
          // get label and label weights
          const int is_pos = is_pos_(label_[i]);
          const int label = label_val_[is_pos];
          const double label_weight = label_weights_[is_pos];
          // calculate gradients and hessians
          const double response = -label * sigmoid_ / (1.0f + std::exp(label * sigmoid_ * score[i]));
          const double abs_response = fabs(response);
          gradients[i] = static_cast<score_t>(response * label_weight);
          hessians[i] = static_cast<score_t>(abs_response * (sigmoid_ - abs_response) * label_weight);
        }
      }
      else
      {
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
        for (data_size_t i = 0; i < num_data_; ++i)
        {
          // get label and label weights
          const int is_pos = is_pos_(label_[i]);
          const int label = label_val_[is_pos];
          const double label_weight = label_weights_[is_pos];
          // calculate gradients and hessians
          const double response = -label * sigmoid_ / (1.0f + std::exp(label * sigmoid_ * score[i]));
          const double abs_response = fabs(response);
          gradients[i] = static_cast<score_t>(response * label_weight * weights_[i]);
          hessians[i] = static_cast<score_t>(abs_response * (sigmoid_ - abs_response) * label_weight * weights_[i]);
        }
      }
    }

    // implement custom average to boost from (if enabled among options)
    double BoostFromScore(int) const override
    {
      double suml = 0.0f;
      double sumw = 0.0f;
      if (weights_ != nullptr)
      {
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+ : suml, sumw) if (!deterministic_)
        for (data_size_t i = 0; i < num_data_; ++i)
        {
          suml += is_pos_(label_[i]) * weights_[i];
          sumw += weights_[i];
        }
      }
      else
      {
        sumw = static_cast<double>(num_data_);
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+ : suml) if (!deterministic_)
        for (data_size_t i = 0; i < num_data_; ++i)
        {
          suml += is_pos_(label_[i]);
        }
      }
      if (Network::num_machines() > 1)
      {
        suml = Network::GlobalSyncUpBySum(suml);
        sumw = Network::GlobalSyncUpBySum(sumw);
      }
      double pavg = suml / sumw;
      pavg = std::min(pavg, 1.0 - kEpsilon);
      pavg = std::max<double>(pavg, kEpsilon);
      double initscore = std::log(pavg / (1.0f - pavg)) / sigmoid_;
      Log::Info("[%s:%s]: pavg=%f -> initscore=%f", GetName(), __func__, pavg, initscore);
      return initscore;
    }

    bool ClassNeedTrain(int /*class_id*/) const override
    {
      return need_train_;
    }

    const char *GetName() const override
    {
      return "binary";
    }

    void ConvertOutput(const double *input, double *output) const override
    {
      output[0] = 1.0f / (1.0f + std::exp(-sigmoid_ * input[0]));
    }

    std::string ToString() const override
    {
      std::stringstream str_buf;
      str_buf << GetName() << " ";
      str_buf << "sigmoid:" << sigmoid_;
      return str_buf.str();
    }

    bool SkipEmptyClass() const override { return true; }

    bool NeedAccuratePrediction() const override { return false; }

    data_size_t NumPositiveData() const override { return num_pos_data_; }

  protected:
    /*! \brief Number of data */
    data_size_t num_data_;
    /*! \brief Number of positive samples */
    data_size_t num_pos_data_;
    /*! \brief Pointer of label */
    const label_t *label_;
    /*! \brief True if using unbalance training */
    bool is_unbalance_;
    /*! \brief Sigmoid parameter */
    double sigmoid_;
    /*! \brief Values for positive and negative labels */
    int label_val_[2];
    /*! \brief Weights for positive and negative labels */
    double label_weights_[2];
    /*! \brief Weights for data */
    const label_t *weights_;
    double scale_pos_weight_;
    std::function<bool(label_t)> is_pos_;
    bool need_train_;
    const bool deterministic_;
  };

  class BinaryExponentialFamilyLoss : public ObjectiveFunction
  {
  public:
    explicit BinaryExponentialFamilyLoss(const Config &config,
                                         std::function<bool(label_t)> is_pos = nullptr)
        : deterministic_(config.deterministic)
    {

      auto const link_name = config.exponential_family_link;
      if (link_name == "canonical" || link_name == "logit")
      {
        link_ = std::shared_ptr<LogitLink>(new LogitLink());
      }
      else if (link_name == "probit")
      {
        link_ = std::shared_ptr<ProbitLink>(new ProbitLink());
      }
      else if (link_name == "cauchit")
      {
        link_ = std::shared_ptr<CauchitLink>(new CauchitLink());
      }
      else if (link_name == "cloglog")
      {
        link_ = std::shared_ptr<CLogLogLink>(new CLogLogLink());
      }
      else if (link_name == "loglog")
      {
        link_ = std::shared_ptr<LogLogLink>(new LogLogLink());
      }
      else
      {
        Log::Fatal("Unknown expenential binary link function %s", link_name.c_str());
      }

      auto const distribution_name = config.exponential_family_distribution;
      if (distribution_name == "bernoulli")
      {
        distribution_ = std::unique_ptr<Bernoulli>(new Bernoulli(link_));
      }
      else
      {
        Log::Fatal("Unknown expenential binary distribution %s", distribution_name.c_str());
      }

      is_unbalance_ = config.is_unbalance;
      scale_pos_weight_ = static_cast<double>(config.scale_pos_weight);
      if (is_unbalance_ && std::fabs(scale_pos_weight_ - 1.0f) > 1e-6)
      {
        Log::Fatal("Cannot set is_unbalance and scale_pos_weight at the same time");
      }
      is_pos_ = is_pos;
      if (is_pos_ == nullptr)
      {
        is_pos_ = [](label_t label)
        { return label > 0; };
      }
    }

    explicit BinaryExponentialFamilyLoss(const std::vector<std::string> &strs)
        : deterministic_(false)
    {
      std::string distribution_name = "";
      std::string link_name = "";
      for (auto str : strs)
      {
        auto tokens = Common::Split(str.c_str(), ':');
        if (tokens.size() == 2)
        {
          if (tokens[0] == std::string("distribution"))
          {
            distribution_name = tokens[1].c_str();
          }
          else if (tokens[0] == std::string("link"))
          {
            link_name = tokens[1].c_str();
          }
        }
      }

      if (link_name == "canonical" || link_name == "logit")
      {
        link_ = std::shared_ptr<LogitLink>(new LogitLink());
      }
      else if (link_name == "probit")
      {
        link_ = std::shared_ptr<ProbitLink>(new ProbitLink());
      }
      else if (link_name == "cauchit")
      {
        link_ = std::shared_ptr<CauchitLink>(new CauchitLink());
      }
      else if (link_name == "cloglog")
      {
        link_ = std::shared_ptr<CLogLogLink>(new CLogLogLink());
      }
      else if (link_name == "loglog")
      {
        link_ = std::shared_ptr<LogLogLink>(new LogLogLink());
      }
      else
      {
        Log::Fatal("Unknown expenential binary link function %s", link_name.c_str());
      }

      if (distribution_name == "bernoulli")
      {
        distribution_ = std::unique_ptr<Bernoulli>(new Bernoulli(link_));
      }
      else
      {
        Log::Fatal("Unknown expenential binary distribution %s", distribution_name.c_str());
      }
    }

    ~BinaryExponentialFamilyLoss() {}

    void Init(const Metadata &metadata, data_size_t num_data) override
    {
      num_data_ = num_data;
      label_ = metadata.label();
      weights_ = metadata.weights();
      data_size_t cnt_positive = 0;
      data_size_t cnt_negative = 0;
// count for positive and negative samples
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+ : cnt_positive, cnt_negative)
      for (data_size_t i = 0; i < num_data_; ++i)
      {
        if (is_pos_(label_[i]))
        {
          ++cnt_positive;
        }
        else
        {
          ++cnt_negative;
        }
      }
      num_pos_data_ = cnt_positive;
      if (Network::num_machines() > 1)
      {
        cnt_positive = Network::GlobalSyncUpBySum(cnt_positive);
        cnt_negative = Network::GlobalSyncUpBySum(cnt_negative);
      }
      need_train_ = true;
      if (cnt_negative == 0 || cnt_positive == 0)
      {
        Log::Warning("Contains only one class");
        // not need to boost.
        need_train_ = false;
      }
      Log::Info("Number of positive: %d, number of negative: %d", cnt_positive, cnt_negative);
      // weight for label
      label_weights_[0] = 1.0f;
      label_weights_[1] = 1.0f;
      // if using unbalance, change the labels weight
      if (is_unbalance_ && cnt_positive > 0 && cnt_negative > 0)
      {
        if (cnt_positive > cnt_negative)
        {
          label_weights_[1] = 1.0f;
          label_weights_[0] = static_cast<double>(cnt_positive) / cnt_negative;
        }
        else
        {
          label_weights_[1] = static_cast<double>(cnt_negative) / cnt_positive;
          label_weights_[0] = 1.0f;
        }
      }
      label_weights_[1] *= scale_pos_weight_;
    }

    void GetGradients(const double *score, score_t *gradients, score_t *hessians) const override
    {
      if (!need_train_)
      {
        return;
      }
      if (weights_ == nullptr)
      {
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
        for (data_size_t i = 0; i < num_data_; ++i)
        {
          // // get label and label weights
          const int is_pos = is_pos_(label_[i]);
          const double label_weight = label_weights_[is_pos];
          // calculate gradients and hessians
          gradients[i] = static_cast<score_t>(distribution_->NegativeLogLikelihoodFirstDerivative(label_[i], score[i])) * label_weight;
          hessians[i] = static_cast<score_t>(distribution_->NegativeLogLikelihoodSecondDerivative(label_[i], score[i])) * label_weight;
        }
      }
      else
      {
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
        for (data_size_t i = 0; i < num_data_; ++i)
        {
          // get label and label weights
          const int is_pos = is_pos_(label_[i]);
          const double label_weight = label_weights_[is_pos];
          // calculate gradients and hessians
          gradients[i] = static_cast<score_t>(distribution_->NegativeLogLikelihoodFirstDerivative(label_[i], score[i])) * label_weight * weights_[i];
          hessians[i] = static_cast<score_t>(distribution_->NegativeLogLikelihoodSecondDerivative(label_[i], score[i])) * label_weight * weights_[i];
        }
      }
    }

    // implement custom average to boost from (if enabled among options)
    double BoostFromScore(int) const override
    {
      double suml = 0.0f;
      double sumw = 0.0f;
      if (weights_ != nullptr)
      {
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+ : suml, sumw) if (!deterministic_)
        for (data_size_t i = 0; i < num_data_; ++i)
        {
          suml += is_pos_(label_[i]) * weights_[i];
          sumw += weights_[i];
        }
      }
      else
      {
        sumw = static_cast<double>(num_data_);
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+ : suml) if (!deterministic_)
        for (data_size_t i = 0; i < num_data_; ++i)
        {
          suml += is_pos_(label_[i]);
        }
      }
      if (Network::num_machines() > 1)
      {
        suml = Network::GlobalSyncUpBySum(suml);
        sumw = Network::GlobalSyncUpBySum(sumw);
      }
      double pavg = suml / sumw;
      pavg = std::min(pavg, 1.0 - kEpsilon);
      pavg = std::max<double>(pavg, kEpsilon);
      double initscore = link_->Function(pavg);
      Log::Info("[%s:%s]: pavg=%f -> initscore=%f", GetName(), __func__, pavg, initscore);
      return initscore;
    }

    bool ClassNeedTrain(int /*class_id*/) const override
    {
      return need_train_;
    }

    const char *GetName() const override
    {
      return "exponential_family";
    }

    void ConvertOutput(const double *input, double *output) const override
    {
      output[0] = link_->Inverse(input[0]);
    }

    std::string ToString() const override
    {
      std::stringstream str_buf;
      str_buf << GetName() << " ";
      str_buf << "distribution:" << distribution_->GetName() << " ";
      str_buf << "link:" << link_->GetName();
      return str_buf.str();
    }

    bool SkipEmptyClass() const override { return true; }

    bool NeedAccuratePrediction() const override { return false; }

    data_size_t NumPositiveData() const override { return num_pos_data_; }

  protected:
    /*! \brief Number of data */
    data_size_t num_data_;
    /*! \brief Number of positive samples */
    data_size_t num_pos_data_;
    /*! \brief Pointer of label */
    const label_t *label_;
    /*! \brief True if using unbalance training */
    bool is_unbalance_;
    /*! \brief Weights for positive and negative labels */
    double label_weights_[2];
    /*! \brief Weights for data */
    const label_t *weights_;
    double scale_pos_weight_;
    std::function<bool(label_t)> is_pos_;
    bool need_train_;
    const bool deterministic_;
    // ExponentialFamilyDistribution* distribution_;
    // ExponentialFamilyLink* link_;
    std::unique_ptr<ExponentialFamilyDistribution> distribution_;
    std::shared_ptr<ExponentialFamilyLink> link_;
  };

} // namespace LightGBM
#endif // LightGBM_OBJECTIVE_BINARY_OBJECTIVE_HPP_
