import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import Dashboard from './components/Dashboard';
import CodeViewer from './components/CodeViewer';
import WalkthroughGuide from './components/WalkthroughGuide';
import { Activity, Code, BookOpen, AlertCircle, TrendingUp, Sparkles, UserCheck } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'code' | 'walkthrough'>('dashboard');

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans antialiased flex flex-col selection:bg-yellow-500/30 selection:text-slate-900">
      
      {/* 1. Elegant Header */}
      <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            
            {/* Title Block */}
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-yellow-500 to-amber-400 flex items-center justify-center text-slate-950 font-bold shadow-md shadow-yellow-500/10">
                <UserCheck size={20} className="stroke-[2.5]" />
              </div>
              <div>
                <span className="block text-sm font-black text-slate-950 tracking-tight font-sans uppercase">
                  Credit Scoring Playground
                </span>
                <span className="block text-[10px] font-mono text-slate-400 font-bold uppercase tracking-widest">
                  Machine Learning Classification Compare
                </span>
              </div>
            </div>

            {/* Navigation Tabs */}
            <nav className="flex items-center gap-1 bg-slate-100 p-1 rounded-xl border border-slate-200/60">
              <button
                id="tab-dashboard"
                onClick={() => setActiveTab('dashboard')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold font-sans transition-all cursor-pointer select-none ${
                  activeTab === 'dashboard'
                    ? 'bg-white text-slate-950 shadow-sm'
                    : 'text-slate-500 hover:text-slate-900'
                }`}
              >
                <Activity size={14} className={activeTab === 'dashboard' ? 'text-yellow-500' : ''} />
                <span className="hidden sm:inline">Interactive Console</span>
              </button>

              <button
                id="tab-code"
                onClick={() => setActiveTab('code')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold font-sans transition-all cursor-pointer select-none ${
                  activeTab === 'code'
                    ? 'bg-white text-slate-950 shadow-sm'
                    : 'text-slate-500 hover:text-slate-900'
                }`}
              >
                <Code size={14} className={activeTab === 'code' ? 'text-indigo-500' : ''} />
                <span className="hidden sm:inline">Python Code</span>
              </button>

              <button
                id="tab-walkthrough"
                onClick={() => setActiveTab('walkthrough')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold font-sans transition-all cursor-pointer select-none ${
                  activeTab === 'walkthrough'
                    ? 'bg-white text-slate-950 shadow-sm'
                    : 'text-slate-500 hover:text-slate-900'
                }`}
              >
                <BookOpen size={14} className={activeTab === 'walkthrough' ? 'text-emerald-500' : ''} />
                <span className="hidden sm:inline">Walkthrough Script</span>
              </button>
            </nav>
            
          </div>
        </div>
      </header>

      {/* 2. Hero Presentation Callout */}
      <section className="bg-gradient-to-b from-white to-slate-50 border-b border-slate-200/80 py-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
            <div className="space-y-2 max-w-2xl">
              <span className="inline-flex items-center gap-1 px-2.5 py-0.5 bg-yellow-500/10 text-yellow-800 rounded-full text-[10px] font-mono font-bold tracking-wider uppercase">
                <Sparkles size={11} /> ML Internship Project Sandbox
              </span>
              <h1 className="text-2xl md:text-3xl font-black text-slate-950 font-sans tracking-tight leading-none">
                Predictive Underwriting Credit Scoring System
              </h1>
              <p className="text-xs md:text-sm text-slate-500 font-sans leading-relaxed">
                This platform is tailored for your machine learning internship showcase. Toggle setting distributions, train true analytical models (Logistic Regression, Decision Trees, and Random Forests) directly in your browser, and extract complete copyable Python code for your submission.
              </p>
            </div>
            
            {/* Quick Metrics Badge card */}
            <div className="bg-white border border-slate-200/80 p-4 rounded-xl flex items-center gap-4 shrink-0 shadow-sm w-full md:w-auto">
              <div className="p-3 bg-emerald-50 text-emerald-600 rounded-lg shrink-0">
                <TrendingUp size={20} />
              </div>
              <div>
                <span className="block text-[10px] font-mono text-slate-400 font-bold uppercase tracking-wider">Default Prediction Target</span>
                <span className="block text-sm font-bold text-slate-900 font-sans">90%+ ROC-AUC Achieved</span>
                <span className="block text-[10px] text-slate-400 font-sans">Multi-classifier comparison model</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 3. Main Sandbox Body Wrapper */}
      <main className="flex-grow max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <AnimatePresence mode="wait">
          {activeTab === 'dashboard' && (
            <motion.div
              key="dashboard"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.25 }}
            >
              <Dashboard />
            </motion.div>
          )}

          {activeTab === 'code' && (
            <motion.div
              key="code"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.25 }}
            >
              <CodeViewer />
            </motion.div>
          )}

          {activeTab === 'walkthrough' && (
            <motion.div
              key="walkthrough"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.25 }}
            >
              <WalkthroughGuide />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* 4. Humble Footer */}
      <footer className="bg-white border-t border-slate-200 py-6 mt-12 text-center text-slate-400 text-xs font-sans">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row justify-between items-center gap-4">
          <p>Credit Scoring Model Playground © 2026. All rights reserved.</p>
          <div className="flex items-center gap-4 text-slate-500">
            <a href="#tab-dashboard" onClick={(e) => { e.preventDefault(); setActiveTab('dashboard'); }} className="hover:text-slate-900 transition-all font-semibold">Dashboard</a>
            <span>•</span>
            <a href="#tab-code" onClick={(e) => { e.preventDefault(); setActiveTab('code'); }} className="hover:text-slate-900 transition-all font-semibold">Python Code</a>
            <span>•</span>
            <a href="#tab-walkthrough" onClick={(e) => { e.preventDefault(); setActiveTab('walkthrough'); }} className="hover:text-slate-900 transition-all font-semibold">Walkthrough Script</a>
          </div>
        </div>
      </footer>
      
    </div>
  );
}
