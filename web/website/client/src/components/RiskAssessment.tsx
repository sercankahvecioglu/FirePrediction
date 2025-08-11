import React from 'react';

interface RiskAssessmentData {
  score: number;
  level: string;
  confidence: number;
}

interface RiskAssessmentProps {
  data: RiskAssessmentData;
}

const RiskAssessment: React.FC<RiskAssessmentProps> = ({ data }) => {
  const getRiskColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'çok yüksek':
      case 'very high':
        return 'text-red-400';
      case 'yüksek':
      case 'high':
        return 'text-orange-400';
      case 'orta':
      case 'medium':
        return 'text-yellow-400';
      case 'orta-düşük':
      case 'medium-low':
        return 'text-blue-400';
      case 'düşük':
      case 'low':
        return 'text-green-400';
      default:
        return 'text-primary-400';
    }
  };

  const getRiskBgColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'çok yüksek':
      case 'very high':
        return 'bg-red-900/20';
      case 'yüksek':
      case 'high':
        return 'bg-orange-900/20';
      case 'orta':
      case 'medium':
        return 'bg-yellow-900/20';
      case 'orta-düşük':
      case 'medium-low':
        return 'bg-blue-900/20';
      case 'düşük':
      case 'low':
        return 'bg-green-900/20';
      default:
        return 'bg-primary-900/20';
    }
  };

  const riskCards = [
    {
      title: 'Risk Skoru',
      value: `${data.score.toFixed(1)}/100`,
      color: getRiskColor(data.level),
      bgColor: getRiskBgColor(data.level),
      borderColor: 'border-primary-500/30'
    },
    {
      title: 'Risk Seviyesi',
      value: data.level,
      color: getRiskColor(data.level),
      bgColor: getRiskBgColor(data.level),
      borderColor: 'border-primary-500/30'
    },
    {
      title: 'Güven Oranı',
      value: `${(data.confidence * 100).toFixed(1)}%`,
      color: 'text-green-400',
      bgColor: 'bg-green-900/20',
      borderColor: 'border-green-500/30'
    }
  ];

  return (
    <div className="space-y-8">
      <h2 className="text-3xl font-bold text-center text-white mb-8">
        AI Risk Değerlendirmesi
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
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

      {/* Risk Score Progress Bar */}
      <div className="card">
        <h3 className="text-xl font-semibold text-white mb-4">Risk Skoru</h3>
        <div className="space-y-4">
          <div className="flex justify-between text-sm text-gray-400">
            <span>Düşük Risk</span>
            <span>Yüksek Risk</span>
          </div>
          <div className="w-full bg-dark-700 rounded-full h-4">
            <div 
              className="bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 h-4 rounded-full transition-all duration-500"
              style={{ width: `${data.score}%` }}
            />
          </div>
          <div className="text-center text-sm text-gray-400">
            Mevcut Risk: {data.score.toFixed(1)}/100
          </div>
        </div>
      </div>
    </div>
  );
};

export default RiskAssessment; 