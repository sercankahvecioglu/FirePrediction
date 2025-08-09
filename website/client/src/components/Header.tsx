import React from 'react';
import { Flame } from 'lucide-react';

const Header: React.FC = () => {
  return (
    <header className="bg-dark-800 border-b border-dark-700">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center space-x-2">
            <Flame className="w-8 h-8 text-primary-400" />
            <span className="text-xl font-bold text-white">FlameSentinels</span>
          </div>

          {/* Navigation */}
          <nav className="hidden md:flex items-center space-x-8">
            <a href="#" className="text-gray-300 hover:text-white transition-colors">
              Home
            </a>
            <a href="#" className="text-gray-300 hover:text-white transition-colors">
              About
            </a>
            <a href="#" className="text-gray-300 hover:text-white transition-colors">
              Documentation
            </a>
            <a href="#" className="text-gray-300 hover:text-white transition-colors">
              Contact
            </a>
          </nav>

          {/* Mobile menu button */}
          <button className="md:hidden text-gray-300 hover:text-white">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header; 