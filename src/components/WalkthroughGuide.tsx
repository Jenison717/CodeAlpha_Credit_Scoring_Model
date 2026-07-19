import React, { useState } from 'react';
import { Play, FileText, CheckCircle, HelpCircle, Layers, TrendingUp, Sparkles, AlertCircle, Quote } from 'lucide-react';

interface WalkthroughStep {
  id: number;
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  objective: string;
  whyItMatters: string;
  interviewPrep: string;
  narratorScript: string;
}

export default function WalkthroughGuide() {
  const [activeStep, setActiveStep] = useState(0);

  const steps: WalkthroughStep[] = [
    {
      id: 1,
      title: "Problem Definition & Dataset",
      subtitle: "Understanding Credit Default Risk",
      icon: <FileText size={18} />,
      objective: "Predict whether an individual is a Good or Bad credit risk (binary classification) based on historical financial attributes: income, age, debt-to-income (DTI), payment history score, loan amount, and employment status.",
      whyItMatters: "Credit scoring is one of the most classic and high-impact applications of ML in finance. A bank needs to minimize default rate (minimizing False Positives—lending to bad credit clients) while maximizing business revenue (minimizing False Negatives—denying good credit clients).",
      interviewPrep: "Our target variable, credit_risk, is binary: '1' represents a low-risk borrower (Good Risk) who pays on time, and '0' represents a high-risk borrower (Bad Risk) likely to default. This is a binary classification problem.",
      narratorScript: "Hi everyone! In this video walkthrough, I am excited to present my machine learning Credit Scoring project. The goal here is to predict individual creditworthiness—that is, whether a person is a good or bad credit risk. We will be using six key columns from our dataset: income, age, debt-to-income ratio, payment history, loan amount, and categorical employment status. We will train and compare three distinct classification algorithms: Logistic Regression, Decision Tree, and Random Forest, to see which model delivers the most reliable performance for financial underwriting."
    },
    {
      id: 2,
      title: "Data Preprocessing Pipeline",
      subtitle: "Imputation, Encoding & Scaling",
      icon: <Layers size={18} />,
      objective: "Clean and prepare the raw data for consumption. This involves median imputation for missing numerical values, one-hot encoding of categorical employment status, and standard scaling (Z-score scaling) of continuous inputs.",
      whyItMatters: "Raw data is often noisy, incomplete, or formatted in ways machine learning algorithms can't handle directly. For instance, distance-based or coefficient-based models like Logistic Regression get biased if one feature has huge numbers (e.g., income in thousands) compared to smaller metrics (e.g., age or DTI).",
      interviewPrep: "Crucial Insight: Feature scaling is MANDATORY for Logistic Regression because it optimizes weights using Gradient Descent; uneven scales distort gradient updates, slowing or preventing convergence. However, scaling is NOT necessary for Decision Trees or Random Forests, because tree-based models split nodes based on order thresholds, which are scale-invariant.",
      narratorScript: "Before feeding our data into any algorithm, we must preprocess it. First, to simulate a real-world scenario where data is often incomplete, I added a few artificial missing values. We resolve this by imputing the median value for missing numbers. Next, we apply One-Hot Encoding to our categorical feature, 'employment_status', dividing it into binary columns so our equations can compute it. Finally, we apply Z-score standardization to all continuous features. This transforms the features to have a mean of 0 and a standard deviation of 1. This step is absolutely critical for Logistic Regression so that larger scale features, like annual income, don't overwhelm smaller features like age during weight updates."
    },
    {
      id: 3,
      title: "Exploratory Data Analysis (EDA)",
      subtitle: "Uncovering Feature Dynamics",
      icon: <TrendingUp size={18} />,
      objective: "Visualize the underlying distributions and relationships in the credit data using correlation matrices and distribution charts.",
      whyItMatters: "EDA gives us statistical intuition. It tells us which variables are strongly tied to default rates and lets us check for issues like multi-collinearity (e.g. if loan amount is perfectly correlated with income), which would destabilize model parameters.",
      interviewPrep: "Correlation is calculated using Pearson's coefficient. A score close to +1.0 indicates a strong positive relationship, -1.0 a strong negative relationship, and 0 indicates no linear correlation. In our heatmap, look out for features strongly correlated with 'credit_risk', like payment history, which usually serves as the strongest predictor.",
      narratorScript: "Next, I performed Exploratory Data Analysis. We look at the Class Distribution of our target first, which tells us if we have class imbalance. In our baseline scenario, we have a healthy balance of about 75% good risks and 25% bad risks. Then we plot a Pearson correlation heatmap. Here we see that 'payment_history' has a strong positive correlation with credit risk, meaning people with high timeliness are highly likely to be classified as good risks. On the other hand, 'debt' shows a strong negative correlation, indicating that a higher debt ratio severely increases credit default risks. Understanding these correlations helps validate that our data aligns with real-world economic logic."
    },
    {
      id: 4,
      title: "Model Architecture & Training",
      subtitle: "Comparing Three Diverse Algorithms",
      icon: <Sparkles size={18} />,
      objective: "Split our preprocessed dataset into train (e.g., 70% or 80%) and test partitions, then train: Logistic Regression (linear boundaries), a Single Decision Tree (hierarchical boundaries), and a Random Forest (ensemble bagging).",
      whyItMatters: "Using a diverse set of algorithms helps us trade off interpretability and accuracy. Logistic Regression is transparent and tells us the precise impact of each unit change. Decision Trees are intuitive to visualize. Random Forests are highly robust, nonlinear models that prevent overfitting.",
      interviewPrep: "Bootstrap sampling means selecting rows with replacement. Feature subsampling means that at each node split, the Random Forest only considers a random subset of features. These two bagging techniques ensure the individual trees are decorrelated, so when we average their predictions, the model's variance is drastically minimized.",
      narratorScript: "Now for the core machine learning phase: we split our dataset into a training set and a testing set. This ensures we evaluate our models on completely unseen data to test for generalizability. We then train three classification algorithms. First, Logistic Regression, which fits a linear decision boundary using a sigmoid probability function. Second, a Single Decision Tree, which recursively segments data into binary nodes by finding thresholds that maximize Gini Impurity reduction. Third, we train a Random Forest. Random Forest is an ensemble method that builds an array of independent decision trees, each trained on a unique bootstrap sample of the data, and aggregates their results to deliver high-precision predictions."
    },
    {
      id: 5,
      title: "Evaluation Metrics",
      subtitle: "Beyond Simple Accuracy",
      icon: <CheckCircle size={18} />,
      objective: "Evaluate the model's test predictions using five core metrics: Accuracy, Precision, Recall, F1-Score, and Receiver Operating Characteristic Area Under the Curve (ROC-AUC).",
      whyItMatters: "Accuracy can be extremely misleading in credit risk. If 95% of customers have good credit, a lazy model that predicts 'good risk' for everyone achieves 95% accuracy but completely fails to detect the 5% defaults. We need Precision (of all predicted good risks, how many are actually safe) and Recall (of all actually safe customers, how many did we approve).",
      interviewPrep: "The Receiver Operating Characteristic (ROC) curve plots True Positive Rate (Sensitivity) on the Y-axis against False Positive Rate (1 - Specificity) on the X-axis for every possible classification threshold. The Area Under this Curve (AUC) measures the model's overall ability to distinguish between the two classes, ranging from 0.5 (random guessing) to 1.0 (perfect classification).",
      narratorScript: "Evaluating classification models in fintech requires looking beyond simple accuracy. We calculate Precision, which is the ratio of true safe loans over all loans we predicted as safe. We calculate Recall, which shows how many of the actual safe borrowers we successfully approved. To get a single balanced harmonic mean of these two, we look at the F1-Score. Finally, we plot the ROC Curve. The ROC curve is generated by sweeping our classification threshold from 0 to 1, mapping out the trade-off between our True Positive Rate and False Positive Rate. The Area Under the Curve, or AUC, is our primary benchmark: a score of 1 represents perfect sorting, while 0.5 represents a model that is no better than a coin toss."
    },
    {
      id: 6,
      title: "Model Recommendation",
      subtitle: "Interpreting Results & Next Steps",
      icon: <Quote size={18} />,
      objective: "Compare the models on a final scoreboard, analyze feature importances, and issue an underwriting recommendation based on statistical findings.",
      whyItMatters: "A junior analyst presents raw numbers, but an elite ML intern presents actionable business decisions. We must explain why the best model won and how it can be controlled in a real lending system.",
      interviewPrep: "Why does Random Forest win? It captures complex non-linear combinations (like 'if income is high AND debt is high, it is still riskier than medium income and low debt') that a linear model misses, while avoiding the overfitting traps of a single, deep decision tree.",
      narratorScript: "Reviewing our final comparison table, we can clearly see the performance trade-offs. While Logistic Regression is fast and highly interpretable, and our Single Decision Tree is incredibly simple to map out, the Random Forest model achieves the highest ROC-AUC and F1-Score. In a lending environment, Random Forest is the superior choice. It captures complex, non-linear feature interactions, such as how debt-to-income limits behave differently across various age brackets. By using Random Forest, a financial institution can maximize their lending approvals while safely shielding themselves from default risks. This concludes my model comparison! Thank you so much for watching, and I am happy to take any questions."
    }
  ];

  const current = steps[activeStep];

  return (
    <div id="walkthrough-guide-container" className="grid grid-cols-1 lg:grid-cols-12 gap-8">
      {/* Sidebar Navigation */}
      <div className="lg:col-span-4 space-y-3">
        <div className="bg-slate-900 text-slate-100 p-4 rounded-xl border border-slate-800">
          <h3 className="text-sm font-semibold font-sans flex items-center gap-2">
            <Play className="text-yellow-500 fill-yellow-500/10" size={16} />
            Presentation Walkthrough Guide
          </h3>
          <p className="text-xs text-slate-400 font-sans mt-1">
            Navigate through the pipeline stages to study explanations and copy ready-made video presentation scripts.
          </p>
        </div>

        <div className="space-y-2">
          {steps.map((s, idx) => (
            <button
              id={`walkthrough-step-btn-${idx}`}
              key={s.id}
              onClick={() => setActiveStep(idx)}
              className={`w-full flex items-start gap-3 p-3.5 rounded-xl border transition-all text-left cursor-pointer ${
                activeStep === idx
                  ? 'bg-yellow-500/10 border-yellow-500/30 text-slate-950 dark:text-slate-100'
                  : 'bg-white border-slate-200 hover:border-slate-300 text-slate-600 hover:text-slate-950'
              }`}
            >
              <span className={`p-1.5 rounded-lg shrink-0 ${
                activeStep === idx ? 'bg-yellow-500 text-slate-950' : 'bg-slate-100 text-slate-500'
              }`}>
                {s.icon}
              </span>
              <div>
                <span className="block text-[10px] font-mono text-slate-400 font-semibold uppercase tracking-wider">
                  Step 0{s.id}
                </span>
                <span className="block text-xs font-semibold font-sans">
                  {s.title}
                </span>
                <span className="block text-[11px] text-slate-400 font-sans truncate max-w-[200px]">
                  {s.subtitle}
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Main Study Panel */}
      <div className="lg:col-span-8 space-y-6">
        {/* Step Header */}
        <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <span className="px-2.5 py-1 bg-slate-100 text-slate-600 rounded-md text-[10px] font-mono font-bold tracking-wider uppercase">
              Step 0{current.id} of 0{steps.length}
            </span>
            <span className="text-xs text-slate-400 font-sans flex items-center gap-1">
              <Sparkles size={14} className="text-yellow-500" /> Walkthrough Script Included
            </span>
          </div>
          <h2 className="text-xl font-bold text-slate-950 font-sans">{current.title}</h2>
          <p className="text-sm text-slate-500 font-sans">{current.subtitle}</p>
        </div>

        {/* Technical Explanations */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white border border-slate-200 rounded-2xl p-5 space-y-3 shadow-sm">
            <h4 className="text-xs font-bold font-mono tracking-wider uppercase text-slate-400 flex items-center gap-1.5">
              <Layers size={14} className="text-indigo-500" />
              1. Step Objective
            </h4>
            <p className="text-xs text-slate-600 font-sans leading-relaxed">
              {current.objective}
            </p>
          </div>

          <div className="bg-white border border-slate-200 rounded-2xl p-5 space-y-3 shadow-sm">
            <h4 className="text-xs font-bold font-mono tracking-wider uppercase text-slate-400 flex items-center gap-1.5">
              <HelpCircle size={14} className="text-emerald-500" />
              2. Technical Relevance
            </h4>
            <p className="text-xs text-slate-600 font-sans leading-relaxed">
              {current.whyItMatters}
            </p>
          </div>
        </div>

        {/* Interview Prep Callout */}
        <div className="bg-amber-50/50 border border-amber-200/60 rounded-2xl p-5 space-y-2.5">
          <h4 className="text-xs font-bold text-amber-800 font-sans flex items-center gap-2">
            <AlertCircle size={16} className="text-amber-600" />
            Interview / Oral Exam Cheat Sheet
          </h4>
          <p className="text-xs text-amber-900 font-sans leading-relaxed">
            {current.interviewPrep}
          </p>
        </div>

        {/* Interactive Narrator Script */}
        <div className="bg-slate-950 text-slate-100 rounded-2xl p-6 border border-slate-800 shadow-xl space-y-4 relative overflow-hidden">
          <div className="absolute right-0 top-0 translate-x-4 -translate-y-4 opacity-[0.03] pointer-events-none">
            <Quote size={150} className="text-white" />
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="flex h-2 w-2 rounded-full bg-red-500 animate-ping"></span>
              <span className="text-[10px] font-mono tracking-widest font-bold text-red-500 uppercase">
                Video Walkthrough Narrator Script
              </span>
            </div>
            <span className="text-xs text-slate-400 font-mono">
              Ready to record
            </span>
          </div>
          <div className="border-l-2 border-yellow-500 pl-4 py-1">
            <p className="text-xs md:text-sm font-sans font-medium leading-relaxed italic text-slate-300">
              "{current.narratorScript}"
            </p>
          </div>
          <div className="flex justify-between items-center text-[10px] text-slate-500 font-sans pt-2 border-t border-slate-800/60">
            <span>Read speed: ~45 seconds</span>
            <span>Tip: Keep your dashboard open and click through the corresponding steps as you read!</span>
          </div>
        </div>
      </div>
    </div>
  );
}
