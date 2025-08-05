import React from 'react';
import { Globe } from 'lucide-react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-dark-800 border-t border-dark-700 mt-16">
      <div className="container mx-auto px-4 py-8">
        <div className="flex flex-col md:flex-row justify-between items-center">
          {/* Copyright */}
          <div className="flex items-center space-x-2 text-gray-400 mb-4 md:mb-0">
            <Globe className="w-4 h-4" />
            <span>© 2024 WildWatch AI. All rights reserved.</span>
          </div>

          {/* Navigation Links */}
          <div className="flex space-x-6">
            <a href="#" className="text-gray-400 hover:text-white transition-colors">
              Terms
            </a>
            <a href="#" className="text-gray-400 hover:text-white transition-colors">
              Privacy
            </a>
            <a href="#" className="text-gray-400 hover:text-white transition-colors">
              Documentation
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer; 