import React from 'react';

interface RiskAssessmentData {
  highRisk: number;
  mediumRisk: number;
  lowRisk: number;
  totalArea: number;
}

interface RiskAssessmentProps {
  data: RiskAssessmentData;
}

const RiskAssessment: React.FC<RiskAssessmentProps> = ({ data }) => {
  const riskCards = [
    {
      title: 'High Risk Areas',
      value: `${data.highRisk}%`,
      color: 'text-red-400',
      bgColor: 'bg-red-900/20',
      borderColor: 'border-red-500/30'
    },
    {
      title: 'Medium Risk',
      value: `${data.mediumRisk}%`,
      color: 'text-yellow-400',
      bgColor: 'bg-yellow-900/20',
      borderColor: 'border-yellow-500/30'
    },
    {
      title: 'Low Risk',
      value: `${data.lowRisk}%`,
      color: 'text-green-400',
      bgColor: 'bg-green-900/20',
      borderColor: 'border-green-500/30'
    },
    {
      title: 'Total Area',
      value: `${data.totalArea} km²`,
      color: 'text-primary-400',
      bgColor: 'bg-primary-900/20',
      borderColor: 'border-primary-500/30'
    }
  ];

  return (
    <div className="space-y-8">
      <h2 className="text-3xl font-bold text-center text-white mb-8">
        Risk Assessment Summary
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {riskCards.map((card, index) => (
          <div 
            key={index}
            className={`
              card border ${card.borderColor} ${card.bgColor}
              hover:scale-105 transition-transform duration-200
            `}
          >
            <h3 className="text-lg font-semibold text-white mb-2">{card.title}</h3>
            <p className={`text-2xl font-bold ${card.color}`}>{card.value}</p>
          </div>
        ))}
      </div>

      {/* Risk Distribution Chart */}
      <div className="card">
        <h3 className="text-xl font-semibold text-white mb-4">Risk Distribution</h3>
        <div className="flex h-8 bg-dark-700 rounded-lg overflow-hidden">
          <div 
            className="bg-red-500 h-full transition-all duration-500"
            style={{ width: `${data.highRisk}%` }}
          />
          <div 
            className="bg-yellow-500 h-full transition-all duration-500"
            style={{ width: `${data.mediumRisk}%` }}
          />
          <div 
            className="bg-green-500 h-full transition-all duration-500"
            style={{ width: `${data.lowRisk}%` }}
          />
        </div>
        <div className="flex justify-between text-sm text-gray-400 mt-2">
          <span>High Risk: {data.highRisk}%</span>
          <span>Medium Risk: {data.mediumRisk}%</span>
          <span>Low Risk: {data.lowRisk}%</span>
        </div>
      </div>
    </div>
  );
};

export default RiskAssessment; 