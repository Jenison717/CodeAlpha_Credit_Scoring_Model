import React, { useState, useEffect } from 'react';
import { generateCreditDataset, calculateSummaryStats, calculateCorrelationMatrix, SCENARIOS } from '../dataGenerator';
import { runMLExperiment } from '../mlEngine';
import { CreditDataRecord, ModelHyperparameters, ModelResults, DatasetScenario } from '../types';
import { Play, RotateCcw, AlertCircle, BarChart3, TrendingUp, Sliders, Settings, Table, HelpCircle, Activity } from 'lucide-react';

export default function Dashboard() {
  // --- STATE ---
  const [scenarioId, setScenarioId] = useState<string>('balanced');
  const [dataSize, setDataSize] = useState<number>(250);
  const [trainRatio, setTrainRatio] = useState<number>(0.8);
  const [threshold, setThreshold] = useState<number>(0.5);

  // Model Hyperparameters
  const [hyperparams, setHyperparams] = useState<ModelHyperparameters>({
    logisticRegression: {
      learningRate: 0.1,
      iterations: 150,
      regularization: 'l2',
      lambda: 0.1
    },
    decisionTree: {
      maxDepth: 5,
      minSamplesSplit: 4
    },
    randomForest: {
      numTrees: 8,
      maxDepth: 6,
      featureSubsampleRatio: 0.7
    }
  });

  // Generated Dataset & Stats
  const [dataset, setDataset] = useState<CreditDataRecord[]>([]);
  const [summaryStats, setSummaryStats] = useState<any>(null);
  const [corrMatrix, setCorrMatrix] = useState<Record<string, Record<string, number>>>({});

  // Model Training Results
  const [results, setResults] = useState<Record<string, ModelResults>>({});
  const [preprocessingLog, setPreprocessingLog] = useState<string[]>([]);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [showLogs, setShowLogs] = useState<boolean>(false);

  // --- HANDLERS ---
  const handleTrainModels = () => {
    setIsTraining(true);
    setTimeout(() => {
      // Find the current scenario object
      const currentScenario = SCENARIOS.find(s => s.id === scenarioId) || SCENARIOS[0];
      
      // Generate dataset (will be consistent per scenario and size)
      const data = generateCreditDataset(currentScenario, dataSize, 42);
      setDataset(data);
      setSummaryStats(calculateSummaryStats(data));
      setCorrMatrix(calculateCorrelationMatrix(data));

      // Run machine learning training & evaluation
      const { results: mlResults, preprocessingLog: logs } = runMLExperiment(
        data,
        hyperparams,
        trainRatio,
        threshold
      );

      setResults(mlResults);
      setPreprocessingLog(logs);
      setIsTraining(false);
    }, 400); // short simulation buffer for active hover UX
  };

  // Run automatically on first render or when scenario / threshold changes
  useEffect(() => {
    handleTrainModels();
  }, [scenarioId, dataSize, trainRatio, threshold]);

  const activeScenario = SCENARIOS.find(s => s.id === scenarioId) || SCENARIOS[0];

  // Helper to reset parameters
  const handleReset = () => {
    setScenarioId('balanced');
    setDataSize(250);
    setTrainRatio(0.8);
    setThreshold(0.5);
    setHyperparams({
      logisticRegression: {
        learningRate: 0.1,
        iterations: 150,
        regularization: 'l2',
        lambda: 0.1
      },
      decisionTree: {
        maxDepth: 5,
        minSamplesSplit: 4
      },
      randomForest: {
        numTrees: 8,
        maxDepth: 6,
        featureSubsampleRatio: 0.7
      }
    });
  };

  // --- RENDER HELPERS ---
  const formatPercentage = (num: number) => `${Math.round(num * 100)}%`;

  // Draw ROC Curve SVG Coordinates
  const renderROCCurve = () => {
    const width = 450;
    const height = 400;
    const padding = 50;

    const mapX = (fpr: number) => padding + fpr * (width - 2 * padding);
    const mapY = (tpr: number) => height - padding - tpr * (height - 2 * padding);

    // Grid lines coordinates
    const gridLines = [0.2, 0.4, 0.6, 0.8];

    // Build SVG paths for our models
    const getPath = (points: { fpr: number; tpr: number }[]) => {
      if (!points || points.length === 0) return '';
      return points
        .map((p, i) => `${i === 0 ? 'M' : 'L'} ${mapX(p.fpr).toFixed(1)} ${mapY(p.tpr).toFixed(1)}`)
        .join(' ');
    };

    const lrPath = results.logisticRegression ? getPath(results.logisticRegression.rocCurve) : '';
    const dtPath = results.decisionTree ? getPath(results.decisionTree.rocCurve) : '';
    const rfPath = results.randomForest ? getPath(results.randomForest.rocCurve) : '';

    // Calculate current metrics coordinates at threshold (closest point to current threshold in TPR/FPR)
    // To make it educational, we find the index of the threshold on the curve or simulate the point
    const getOperatingPoint = (modelRes: ModelResults) => {
      if (!modelRes) return null;
      // Evaluate metrics to map coordinates
      const rec = modelRes.metrics.recall; // Recall = TPR
      // FPR = FP / (FP + TN). Let's fetch TPR/FPR close to current metric evaluation.
      // We can find the closest point in the ROC points array
      let closest = modelRes.rocCurve[0];
      let minDiff = 999;
      modelRes.rocCurve.forEach(p => {
        const diff = Math.abs(p.tpr - rec);
        if (diff < minDiff) {
          minDiff = diff;
          closest = p;
        }
      });
      return closest;
    };

    const lrPoint = results.logisticRegression ? getOperatingPoint(results.logisticRegression) : null;
    const dtPoint = results.decisionTree ? getOperatingPoint(results.decisionTree) : null;
    const rfPoint = results.randomForest ? getOperatingPoint(results.randomForest) : null;

    return (
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto bg-slate-950 rounded-2xl border border-slate-800 text-slate-100 font-sans shadow-inner">
        {/* Title */}
        <text x={width / 2} y={25} textAnchor="middle" className="text-xs font-semibold fill-slate-300 font-sans uppercase tracking-wider">
          ROC Curve Comparison
        </text>

        {/* Diagonal Random Guess Line */}
        <line
          x1={mapX(0)}
          y1={mapY(0)}
          x2={mapX(1)}
          y2={mapY(1)}
          stroke="#475569"
          strokeWidth="1.5"
          strokeDasharray="4 4"
        />

        {/* Grid Lines */}
        {gridLines.map(v => (
          <g key={v}>
            {/* Horizontal Grid */}
            <line
              x1={mapX(0)}
              y1={mapY(v)}
              x2={mapX(1)}
              y2={mapY(v)}
              stroke="#1e293b"
              strokeWidth="1"
            />
            <text x={padding - 10} y={mapY(v) + 4} textAnchor="end" className="text-[10px] font-mono fill-slate-500">
              {v.toFixed(1)}
            </text>

            {/* Vertical Grid */}
            <line
              x1={mapX(v)}
              y1={mapY(0)}
              x2={mapX(v)}
              y2={mapY(1)}
              stroke="#1e293b"
              strokeWidth="1"
            />
            <text x={mapX(v)} y={height - padding + 15} textAnchor="middle" className="text-[10px] font-mono fill-slate-500">
              {v.toFixed(1)}
            </text>
          </g>
        ))}

        {/* Borders and Axes */}
        <line x1={mapX(0)} y1={mapY(0)} x2={mapX(0)} y2={mapY(1)} stroke="#334155" strokeWidth="1.5" />
        <line x1={mapX(0)} y1={mapY(0)} x2={mapX(1)} y2={mapY(0)} stroke="#334155" strokeWidth="1.5" />
        <text x={padding - 10} y={mapY(0) + 4} textAnchor="end" className="text-[10px] font-mono fill-slate-500">0.0</text>
        <text x={padding - 10} y={mapY(1) + 4} textAnchor="end" className="text-[10px] font-mono fill-slate-500">1.0</text>
        <text x={mapX(0)} y={height - padding + 15} textAnchor="middle" className="text-[10px] font-mono fill-slate-500">0.0</text>
        <text x={mapX(1)} y={height - padding + 15} textAnchor="middle" className="text-[10px] font-mono fill-slate-500">1.0</text>

        {/* Axes Labels */}
        <text x={width / 2} y={height - 12} textAnchor="middle" className="text-[10px] font-semibold fill-slate-400 font-sans">
          False Positive Rate (FPR) →
        </text>
        <text
          x={14}
          y={height / 2}
          textAnchor="middle"
          transform={`rotate(-90 14 ${height / 2})`}
          className="text-[10px] font-semibold fill-slate-400 font-sans"
        >
          True Positive Rate (TPR / Recall) →
        </text>

        {/* Plot Model Paths */}
        {lrPath && (
          <path d={lrPath} fill="none" stroke="#60a5fa" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
        )}
        {dtPath && (
          <path d={dtPath} fill="none" stroke="#f59e0b" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
        )}
        {rfPath && (
          <path d={rfPath} fill="none" stroke="#10b981" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
        )}

        {/* Operational Threshold Highlight Lines */}
        {lrPoint && dtPoint && rfPoint && (
          <>
            {/* Draw a vertical line representing our current active operating space */}
            <circle cx={mapX(lrPoint.fpr)} cy={mapY(lrPoint.tpr)} r="5" fill="#60a5fa" stroke="#0f172a" strokeWidth="1.5" />
            <circle cx={mapX(dtPoint.fpr)} cy={mapY(dtPoint.tpr)} r="5" fill="#f59e0b" stroke="#0f172a" strokeWidth="1.5" />
            <circle cx={mapX(rfPoint.fpr)} cy={mapY(rfPoint.tpr)} r="5" fill="#10b981" stroke="#0f172a" strokeWidth="1.5" />
            
            {/* Operating Label details */}
            <rect x={width - 150} y={height - 130} width="115" height="60" rx="6" fill="#1e293b" stroke="#334155" strokeWidth="1" />
            <text x={width - 140} y={height - 114} className="text-[10px] font-semibold fill-slate-300">Operating Point:</text>
            <text x={width - 140} y={height - 98} className="text-[9px] font-sans fill-slate-400">
              Threshold: <tspan className="font-mono text-yellow-500 font-bold">{threshold.toFixed(2)}</tspan>
            </text>
            <text x={width - 140} y={height - 84} className="text-[9px] font-sans fill-slate-400">
              RF Recall/TPR: <tspan className="font-mono text-emerald-400 font-bold">{rfPoint.tpr.toFixed(2)}</tspan>
            </text>
          </>
        )}

        {/* Legend */}
        <g transform={`translate(${padding + 15}, ${padding + 15})`}>
          <rect width="185" height="70" rx="8" fill="#1e293b/80" stroke="#334155" strokeWidth="1" className="backdrop-blur-sm" />
          
          <circle cx="12" cy="15" r="4" fill="#60a5fa" />
          <text x="24" y="19" className="text-[9px] font-semibold fill-slate-300 font-sans">
            Logistic Regression (AUC: {results.logisticRegression?.metrics.auc.toFixed(3)})
          </text>

          <circle cx="12" cy="33" r="4" fill="#f59e0b" />
          <text x="24" y="37" className="text-[9px] font-semibold fill-slate-300 font-sans">
            Decision Tree (AUC: {results.decisionTree?.metrics.auc.toFixed(3)})
          </text>

          <circle cx="12" cy="51" r="4" fill="#10b981" />
          <text x="24" y="55" className="text-[9px] font-semibold fill-slate-300 font-sans">
            Random Forest (AUC: {results.randomForest?.metrics.auc.toFixed(3)})
          </text>
        </g>
      </svg>
    );
  };

  // Class distribution bar chart using basic SVG
  const renderClassDistribution = () => {
    if (!summaryStats) return null;
    const goodVal = summaryStats.goodRisk;
    const badVal = summaryStats.badRisk;
    const total = summaryStats.total;
    const goodPct = summaryStats.goodRiskPercent;
    const badPct = summaryStats.badRiskPercent;

    const maxVal = Math.max(goodVal, badVal);
    const graphWidth = 100; // percent

    return (
      <div id="class-dist-chart" className="space-y-4">
        <h4 className="text-xs font-bold text-slate-500 font-mono tracking-wider uppercase flex items-center gap-1.5">
          <BarChart3 size={14} className="text-indigo-500" />
          Class Target Distribution
        </h4>
        <div className="space-y-3.5 bg-slate-50 p-4 rounded-xl border border-slate-200">
          {/* Good Credit Risk */}
          <div className="space-y-1.5">
            <div className="flex justify-between items-center text-xs">
              <span className="font-semibold text-emerald-700 flex items-center gap-1.5 font-sans">
                <span className="h-2 w-2 rounded-full bg-emerald-500"></span>
                Good Credit Risk (Class 1)
              </span>
              <span className="font-mono text-slate-600">
                {goodVal} / {total} records ({goodPct}%)
              </span>
            </div>
            <div className="w-full bg-slate-200 h-6 rounded-md overflow-hidden relative shadow-inner">
              <div
                className="bg-emerald-500 h-full rounded-md transition-all duration-500"
                style={{ width: `${(goodVal / maxVal) * 100}%` }}
              ></div>
            </div>
          </div>

          {/* Bad Credit Risk */}
          <div className="space-y-1.5">
            <div className="flex justify-between items-center text-xs">
              <span className="font-semibold text-rose-700 flex items-center gap-1.5 font-sans">
                <span className="h-2 w-2 rounded-full bg-rose-500"></span>
                Bad Credit Risk (Class 0)
              </span>
              <span className="font-mono text-slate-600">
                {badVal} / {total} records ({badPct}%)
              </span>
            </div>
            <div className="w-full bg-slate-200 h-6 rounded-md overflow-hidden relative shadow-inner">
              <div
                className="bg-rose-500 h-full rounded-md transition-all duration-500"
                style={{ width: `${(badVal / maxVal) * 100}%` }}
              ></div>
            </div>
          </div>
        </div>
        <p className="text-[10px] text-slate-400 font-sans leading-tight">
          Note: An imbalanced dataset typically favors accuracy scores toward the majority class. Standardizing or stratifying during splits helps maintain realistic recall evaluation.
        </p>
      </div>
    );
  };

  // Correlation Heatmap using custom HTML tables
  const renderHeatmap = () => {
    if (!corrMatrix || Object.keys(corrMatrix).length === 0) return null;
    const keys = ['income', 'age', 'debt', 'payment_history', 'loan_amount', 'credit_risk'];
    const labels: Record<string, string> = {
      income: 'Income',
      age: 'Age',
      debt: 'Debt %',
      payment_history: 'Pay History',
      loan_amount: 'Loan Amt',
      credit_risk: 'Risk Target'
    };

    // Color gradient based on correlation value
    const getCellColor = (val: number) => {
      // +1 is deep green-blue, -1 is deep red, 0 is white
      if (val === 1) return 'bg-slate-200 font-bold text-slate-900 border border-slate-300';
      
      const abs = Math.abs(val);
      if (val > 0) {
        if (abs > 0.6) return 'bg-emerald-500 text-white font-semibold';
        if (abs > 0.4) return 'bg-emerald-400 text-slate-950';
        if (abs > 0.2) return 'bg-emerald-200 text-slate-950';
        return 'bg-emerald-50/50 text-slate-800';
      } else {
        if (abs > 0.6) return 'bg-rose-500 text-white font-semibold';
        if (abs > 0.4) return 'bg-rose-400 text-slate-950';
        if (abs > 0.2) return 'bg-rose-200 text-slate-950';
        return 'bg-rose-50/50 text-slate-800';
      }
    };

    return (
      <div id="correlation-heatmap" className="space-y-4">
        <h4 className="text-xs font-bold text-slate-500 font-mono tracking-wider uppercase flex items-center gap-1.5">
          <TrendingUp size={14} className="text-emerald-500" />
          Pearson Correlation Heatmap
        </h4>
        <div className="overflow-x-auto bg-white border border-slate-200 rounded-xl p-3 shadow-sm">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr>
                <th className="p-1.5 text-[10px] font-mono text-slate-400 font-bold uppercase tracking-wider">Metric</th>
                {keys.map(k => (
                  <th key={k} className="p-1.5 text-[10px] font-mono text-slate-500 font-bold uppercase tracking-wider text-center">
                    {labels[k]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {keys.map(f1 => (
                <tr key={f1} className="border-t border-slate-100">
                  <td className="p-1.5 text-xs font-semibold text-slate-700 font-sans">{labels[f1]}</td>
                  {keys.map(f2 => {
                    const r = corrMatrix[f1]?.[f2] ?? 0;
                    return (
                      <td
                        key={f2}
                        className={`p-2 text-center text-xs font-mono rounded-md transition-all duration-200 select-all ${getCellColor(r)}`}
                        title={`Correlation between ${labels[f1]} and ${labels[f2]}: ${r}`}
                      >
                        {r > 0 && r < 1 ? `+${r.toFixed(2)}` : r.toFixed(2)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="flex justify-between items-center text-[10px] text-slate-400 font-sans">
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-emerald-500"></span> Positive correlation (reduces default risk)
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-rose-500"></span> Negative correlation (elevates default risk)
          </span>
        </div>
      </div>
    );
  };

  // Find the winning score for each metric across models
  const getWinner = (metricKey: 'accuracy' | 'precision' | 'recall' | 'f1Score' | 'auc') => {
    if (!results.logisticRegression || !results.decisionTree || !results.randomForest) return '';
    const lr = results.logisticRegression.metrics[metricKey];
    const dt = results.decisionTree.metrics[metricKey];
    const rf = results.randomForest.metrics[metricKey];

    const max = Math.max(lr, dt, rf);
    if (lr === max) return 'logisticRegression';
    if (dt === max) return 'decisionTree';
    return 'randomForest';
  };

  return (
    <div id="dashboard-layout" className="grid grid-cols-1 xl:grid-cols-12 gap-8">
      {/* 1. Sidebar Control Panel */}
      <div className="xl:col-span-4 space-y-6">
        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-5">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <h3 className="font-bold text-slate-900 font-sans flex items-center gap-2">
              <Sliders className="text-yellow-500" size={18} />
              Simulation Settings
            </h3>
            <button
              id="reset-simulation-btn"
              onClick={handleReset}
              className="text-xs text-slate-400 hover:text-slate-900 flex items-center gap-1 cursor-pointer"
              title="Reset to initial default simulation parameters"
            >
              <RotateCcw size={13} /> Reset
            </button>
          </div>

          {/* Dataset Scenario selection */}
          <div className="space-y-2">
            <label className="text-xs font-bold text-slate-700 font-sans flex items-center gap-1">
              Economic Dataset Scenario
            </label>
            <select
              id="scenario-select"
              value={scenarioId}
              onChange={e => setScenarioId(e.target.value)}
              className="w-full p-2.5 bg-slate-50 border border-slate-200 text-slate-800 rounded-xl text-xs font-sans font-medium focus:ring-2 focus:ring-yellow-500 focus:outline-none"
            >
              {SCENARIOS.map(sc => (
                <option key={sc.id} value={sc.id}>{sc.name}</option>
              ))}
            </select>
            <p className="text-[10px] text-slate-400 font-sans italic">
              {activeScenario.description}
            </p>
          </div>

          {/* Train Test Split slider */}
          <div className="space-y-1.5">
            <div className="flex justify-between items-center text-xs">
              <span className="font-bold text-slate-700 font-sans">Train / Test Split Ratio</span>
              <span className="font-mono text-slate-500 font-semibold">{formatPercentage(trainRatio)} / {formatPercentage(1 - trainRatio)}</span>
            </div>
            <input
              id="train-ratio-slider"
              type="range"
              min="0.5"
              max="0.9"
              step="0.05"
              value={trainRatio}
              onChange={e => setTrainRatio(parseFloat(e.target.value))}
              className="w-full accent-yellow-500 cursor-pointer h-1.5 bg-slate-200 rounded-lg appearance-none"
            />
          </div>

          {/* Classification Threshold */}
          <div className="space-y-1.5">
            <div className="flex justify-between items-center text-xs">
              <span className="font-bold text-slate-700 font-sans">Classification Threshold</span>
              <span className="font-mono text-slate-900 font-bold">{threshold.toFixed(2)}</span>
            </div>
            <input
              id="threshold-slider"
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={threshold}
              onChange={e => setThreshold(parseFloat(e.target.value))}
              className="w-full accent-yellow-500 cursor-pointer h-1.5 bg-slate-200 rounded-lg appearance-none"
            />
            <p className="text-[10px] text-slate-400 font-sans">
              Increasing this requires a higher predicted probability to approve credit (lower Recall, higher Precision).
            </p>
          </div>
        </div>

        {/* Hyperparameter Settings */}
        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-5">
          <h3 className="font-bold text-slate-900 font-sans flex items-center gap-2 border-b border-slate-100 pb-3">
            <Settings className="text-indigo-500" size={18} />
            Model Hyperparameters
          </h3>

          {/* 1. Logistic Regression */}
          <div className="space-y-3 bg-slate-50/50 p-3 rounded-xl border border-slate-100">
            <h4 className="text-xs font-bold text-slate-800 font-sans flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-blue-500"></span>
              Logistic Regression
            </h4>
            <div className="space-y-3">
              <div className="space-y-1">
                <div className="flex justify-between text-[11px] text-slate-500 font-sans">
                  <span>Learning Rate (α)</span>
                  <span className="font-mono">{hyperparams.logisticRegression.learningRate}</span>
                </div>
                <input
                  type="range"
                  min="0.01"
                  max="0.5"
                  step="0.01"
                  value={hyperparams.logisticRegression.learningRate}
                  onChange={e => setHyperparams({
                    ...hyperparams,
                    logisticRegression: { ...hyperparams.logisticRegression, learningRate: parseFloat(e.target.value) }
                  })}
                  className="w-full accent-blue-500 cursor-pointer h-1 bg-slate-200 rounded-lg appearance-none"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-[11px] text-slate-500 font-sans">
                  <span>Iterations</span>
                  <span className="font-mono">{hyperparams.logisticRegression.iterations}</span>
                </div>
                <input
                  type="range"
                  min="50"
                  max="300"
                  step="10"
                  value={hyperparams.logisticRegression.iterations}
                  onChange={e => setHyperparams({
                    ...hyperparams,
                    logisticRegression: { ...hyperparams.logisticRegression, iterations: parseInt(e.target.value) }
                  })}
                  className="w-full accent-blue-500 cursor-pointer h-1 bg-slate-200 rounded-lg appearance-none"
                />
              </div>
            </div>
          </div>

          {/* 2. Decision Tree */}
          <div className="space-y-3 bg-slate-50/50 p-3 rounded-xl border border-slate-100">
            <h4 className="text-xs font-bold text-slate-800 font-sans flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-amber-500"></span>
              Decision Tree
            </h4>
            <div className="space-y-3">
              <div className="space-y-1">
                <div className="flex justify-between text-[11px] text-slate-500 font-sans">
                  <span>Max Depth</span>
                  <span className="font-mono">{hyperparams.decisionTree.maxDepth}</span>
                </div>
                <input
                  type="range"
                  min="2"
                  max="10"
                  step="1"
                  value={hyperparams.decisionTree.maxDepth}
                  onChange={e => setHyperparams({
                    ...hyperparams,
                    decisionTree: { ...hyperparams.decisionTree, maxDepth: parseInt(e.target.value) }
                  })}
                  className="w-full accent-amber-500 cursor-pointer h-1 bg-slate-200 rounded-lg appearance-none"
                />
              </div>
            </div>
          </div>

          {/* 3. Random Forest */}
          <div className="space-y-3 bg-slate-50/50 p-3 rounded-xl border border-slate-100">
            <h4 className="text-xs font-bold text-slate-800 font-sans flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-emerald-500"></span>
              Random Forest (Bagged Ensemble)
            </h4>
            <div className="space-y-3">
              <div className="space-y-1">
                <div className="flex justify-between text-[11px] text-slate-500 font-sans">
                  <span>Number of Trees</span>
                  <span className="font-mono">{hyperparams.randomForest.numTrees}</span>
                </div>
                <input
                  type="range"
                  min="3"
                  max="20"
                  step="1"
                  value={hyperparams.randomForest.numTrees}
                  onChange={e => setHyperparams({
                    ...hyperparams,
                    randomForest: { ...hyperparams.randomForest, numTrees: parseInt(e.target.value) }
                  })}
                  className="w-full accent-emerald-500 cursor-pointer h-1 bg-slate-200 rounded-lg appearance-none"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-[11px] text-slate-500 font-sans">
                  <span>Max Depth</span>
                  <span className="font-mono">{hyperparams.randomForest.maxDepth}</span>
                </div>
                <input
                  type="range"
                  min="2"
                  max="10"
                  step="1"
                  value={hyperparams.randomForest.maxDepth}
                  onChange={e => setHyperparams({
                    ...hyperparams,
                    randomForest: { ...hyperparams.randomForest, maxDepth: parseInt(e.target.value) }
                  })}
                  className="w-full accent-emerald-500 cursor-pointer h-1 bg-slate-200 rounded-lg appearance-none"
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Visualizations and Evaluation Dashboards */}
      <div className="xl:col-span-8 space-y-6">
        {/* ML Scoreboard Table */}
        <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <h3 className="font-bold text-slate-900 font-sans flex items-center gap-2">
              <Table className="text-yellow-500" size={18} />
              Model Comparison Table (Unseen Test Set)
            </h3>
            <span className="text-[10px] bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded font-mono font-bold uppercase">
              Live Testing Results
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500 text-[10px] font-mono font-bold uppercase tracking-wider">
                  <th className="pb-3 text-left">Algorithm</th>
                  <th className="pb-3 text-center">Accuracy</th>
                  <th className="pb-3 text-center">Precision</th>
                  <th className="pb-3 text-center">Recall (TPR)</th>
                  <th className="pb-3 text-center">F1-Score</th>
                  <th className="pb-3 text-center">ROC-AUC</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-xs font-sans">
                {/* 1. Logistic Regression */}
                {results.logisticRegression && (
                  <tr>
                    <td className="py-3.5 font-semibold text-slate-800 flex items-center gap-2">
                      <span className="h-2 w-2 rounded-full bg-blue-500"></span>
                      Logistic Regression
                    </td>
                    <td className={`py-3.5 text-center font-mono ${getWinner('accuracy') === 'logisticRegression' ? 'text-emerald-600 font-bold bg-emerald-50/20' : 'text-slate-600'}`}>
                      {results.logisticRegression.metrics.accuracy.toFixed(3)}
                    </td>
                    <td className={`py-3.5 text-center font-mono ${getWinner('precision') === 'logisticRegression' ? 'text-emerald-600 font-bold bg-emerald-50/20' : 'text-slate-600'}`}>
                      {results.logisticRegression.metrics.precision.toFixed(3)}
                    </td>
                    <td className={`py-3.5 text-center font-mono ${getWinner('recall') === 'logisticRegression' ? 'text-emerald-600 font-bold bg-emerald-50/20' : 'text-slate-600'}`}>
                      {results.logisticRegression.metrics.recall.toFixed(3)}
                    </td>
                    <td className={`py-3.5 text-center font-mono ${getWinner('f1Score') === 'logisticRegression' ? 'text-emerald-600 font-bold bg-emerald-50/20' : 'text-slate-600'}`}>
                      {results.logisticRegression.metrics.f1Score.toFixed(3)}
                    </td>
                    <td className={`py-3.5 text-center font-mono ${getWinner('auc') === 'logisticRegression' ? 'text-emerald-600 font-bold bg-emerald-50/20' : 'text-slate-600'}`}>
                      {results.logisticRegression.metrics.auc.toFixed(3)}
                    </td>
                  </tr>
                )}

                {/* 2. Decision Tree */}
                {results.decisionTree && (
                  <tr>
                    <td className="py-3.5 font-semibold text-slate-800 flex items-center gap-2">
                      <span className="h-2 w-2 rounded-full bg-amber-500"></span>
                      Decision Tree
                    </td>
                    <td className={`py-3.5 text-center font-mono ${getWinner('accuracy') === 'decisionTree' ? 'text-emerald-600 font-bold bg-emerald-50/20' : 'text-slate-600'}`}>
                      {results.decisionTree.metrics.accuracy.toFixed(3)}
                    </td>
                    <td className={`py-3.5 text-center font-mono ${getWinner('precision') === 'decisionTree' ? 'text-emerald-600 font-bold bg-emerald-50/20' : 'text-slate-600'}`}>
                      {results.decisionTree.metrics.precision.toFixed(3)}
                    </td>
                    <td className={`py-3.5 text-center font-mono ${getWinner('recall') === 'decisionTree' ? 'text-emerald-600 font-bold bg-emerald-50/20' : 'text-slate-600'}`}>
                      {results.decisionTree.metrics.recall.toFixed(3)}
                    </td>
                    <td className={`py-3.5 text-center font-mono ${getWinner('f1Score') === 'decisionTree' ? 'text-emerald-600 font-bold bg-emerald-50/20' : 'text-slate-600'}`}>
                      {results.decisionTree.metrics.f1Score.toFixed(3)}
                    </td>
                    <td className={`py-3.5 text-center font-mono ${getWinner('auc') === 'decisionTree' ? 'text-emerald-600 font-bold bg-emerald-50/20' : 'text-slate-600'}`}>
                      {results.decisionTree.metrics.auc.toFixed(3)}
                    </td>
                  </tr>
                )}

                {/* 3. Random Forest */}
                {results.randomForest && (
                  <tr>
                    <td className="py-3.5 font-semibold text-slate-800 flex items-center gap-2">
                      <span className="h-2 w-2 rounded-full bg-emerald-500"></span>
                      Random Forest
                    </td>
                    <td className={`py-3.5 text-center font-mono ${getWinner('accuracy') === 'randomForest' ? 'text-emerald-600 font-bold bg-emerald-50/20' : 'text-slate-600'}`}>
                      {results.randomForest.metrics.accuracy.toFixed(3)}
                    </td>
                    <td className={`py-3.5 text-center font-mono ${getWinner('precision') === 'randomForest' ? 'text-emerald-600 font-bold bg-emerald-50/20' : 'text-slate-600'}`}>
                      {results.randomForest.metrics.precision.toFixed(3)}
                    </td>
                    <td className={`py-3.5 text-center font-mono ${getWinner('recall') === 'randomForest' ? 'text-emerald-600 font-bold bg-emerald-50/20' : 'text-slate-600'}`}>
                      {results.randomForest.metrics.recall.toFixed(3)}
                    </td>
                    <td className={`py-3.5 text-center font-mono ${getWinner('f1Score') === 'randomForest' ? 'text-emerald-600 font-bold bg-emerald-50/20' : 'text-slate-600'}`}>
                      {results.randomForest.metrics.f1Score.toFixed(3)}
                    </td>
                    <td className={`py-3.5 text-center font-mono ${getWinner('auc') === 'randomForest' ? 'text-emerald-600 font-bold bg-emerald-50/20' : 'text-slate-600'}`}>
                      {results.randomForest.metrics.auc.toFixed(3)}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <p className="text-[10px] text-slate-400 font-sans italic">
            Highlighted cells indicate the leading algorithm on the respective evaluation metric. Random Forest typically leads overall due to ensemble variance reduction.
          </p>
        </div>

        {/* Visual Charts Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* ROC Graph */}
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm space-y-4">
            <h3 className="font-bold text-slate-900 font-sans flex items-center gap-2">
              <Activity className="text-yellow-500" size={18} />
              Evaluation: ROC Curves
            </h3>
            {renderROCCurve()}
          </div>

          {/* EDA: Heatmap & Class Distributions */}
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm space-y-6">
            <h3 className="font-bold text-slate-900 font-sans flex items-center gap-2 border-b border-slate-100 pb-3">
              <BarChart3 className="text-yellow-500" size={18} />
              Exploratory Data Analysis
            </h3>
            {renderClassDistribution()}
            <div className="border-t border-slate-100 pt-5">
              {renderHeatmap()}
            </div>
          </div>
        </div>

        {/* Console / Log view */}
        <div className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
          <button
            id="toggle-logs-btn"
            onClick={() => setShowLogs(!showLogs)}
            className="w-full px-6 py-4 flex items-center justify-between text-left bg-slate-50 hover:bg-slate-100 transition-colors cursor-pointer"
          >
            <div className="flex items-center gap-2 text-slate-800">
              <span className="p-1 bg-slate-900 text-yellow-400 font-mono text-[9px] rounded uppercase font-bold">Log</span>
              <h4 className="text-sm font-bold font-sans">View Preprocessing & Pipeline Console Log</h4>
            </div>
            <span className="text-xs text-slate-400 font-sans">
              {showLogs ? 'Hide Console [▲]' : 'Show Console [▼]'}
            </span>
          </button>
          
          {showLogs && (
            <div className="bg-slate-950 p-5 font-mono text-xs text-slate-300 leading-relaxed border-t border-slate-800 max-h-[300px] overflow-y-auto">
              <div className="flex items-center gap-1.5 text-yellow-500 border-b border-slate-800 pb-2 mb-3">
                <span className="h-2 w-2 rounded-full bg-yellow-500 animate-pulse"></span>
                <span>Active Sandbox Preprocessor & Compiler Diagnostics</span>
              </div>
              {preprocessingLog.map((log, index) => (
                <div key={index} className="py-0.5">
                  <span className="text-slate-500 select-none mr-2">[{index+1}]</span>
                  {log}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
