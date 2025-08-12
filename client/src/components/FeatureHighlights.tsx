import React from 'react';
import { Gauge, Clock, Globe } from 'lucide-react';

const FeatureHighlights: React.FC = () => {
  const features = [
    {
      icon: Gauge,
      title: '90.8% Accuracy',
      description: 'High Accuracy'
    },
    {
      icon: Clock,
      title: 'Real-time Analysis',
      description: 'Real Time Analysis'
    },
    {
      icon: Globe,
      title: 'Global Coverage',
      description: 'Global Coverage'
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
      {features.map((feature, index) => (
        <div key={index} className="text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-primary-900/20 rounded-full mb-4">
            <feature.icon className="w-8 h-8 text-primary-400" />
          </div>
          <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
          <p className="text-gray-400">{feature.description}</p>
        </div>
      ))}
    </div>
  );
};

export default FeatureHighlights; 