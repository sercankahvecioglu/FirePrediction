#!/usr/bin/env python3
"""
Startup script to launch both the Telegram bot and the main FastAPI server
"""

import subprocess
import time
import sys
import os
import signal
import threading

def start_telegram_bot():
    """Start the Telegram bot server"""
    print("🤖 Starting Telegram Bot Server...")
    telegram_dir = os.path.join(os.path.dirname(__file__), "utils", "TelegramBot")
    telegram_script = os.path.join(telegram_dir, "server.py")
    
    if not os.path.exists(telegram_script):
        print(f"❌ Telegram bot script not found: {telegram_script}")
        return None
    
    try:
        # Use environment variables for configuration
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        process = subprocess.Popen(
            [sys.executable, telegram_script],
            cwd=telegram_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            text=True,
            env=env
        )
        print(f"✅ Telegram Bot started with PID: {process.pid}")
        return process
    except Exception as e:
        print(f"❌ Error starting Telegram bot: {e}")
        return None

def start_main_server():
    """Start the main FastAPI server"""
    print("🚀 Starting Main FastAPI Server...")
    
    try:
        # Use environment variables for configuration
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        process = subprocess.Popen(
            [sys.executable, "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            text=True,
            env=env
        )
        print(f"✅ Main server started with PID: {process.pid}")
        return process
    except Exception as e:
        print(f"❌ Error starting main server: {e}")
        return None

def monitor_process(process, name, color_code="37"):
    """Monitor a process and print its output with color coding"""
    def read_output():
        while True:
            try:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Add color and timestamp
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"\033[{color_code}m[{timestamp}] [{name}]\033[0m {output.strip()}")
            except Exception as e:
                print(f"❌ Error reading output from {name}: {e}")
                break
    
    # Start thread to read output
    output_thread = threading.Thread(target=read_output, daemon=True)
    output_thread.start()

def main():
    """Main function to start both services"""
    print("🔥 Fire Prediction System - Starting Services...")
    print("=" * 60)
    
    processes = []
    
    try:
        # Start Telegram bot first
        telegram_process = start_telegram_bot()
        if telegram_process:
            processes.append(("Telegram Bot", telegram_process))
            monitor_process(telegram_process, "TELEGRAM", "34")  # Blue
        
        # Wait a bit for Telegram bot to initialize
        time.sleep(5)
        
        # Start main server
        main_process = start_main_server()
        if main_process:
            processes.append(("Main Server", main_process))
            monitor_process(main_process, "MAIN", "32")  # Green
        
        if not processes:
            print("❌ No services could be started!")
            return 1
        
        print("\n" + "=" * 60)
        print("🎉 Services started successfully!")
        print("📊 Main API: http://localhost:5001")
        print("🤖 Telegram Bot API: http://localhost:5002")
        print("📱 Test Telegram: http://localhost:5001/test-telegram-alert")
        print("📋 API Documentation: http://localhost:5001/docs")
        print("\nPress Ctrl+C to stop all services...")
        print("=" * 60)
        
        # Health check loop
        health_check_interval = 30  # seconds
        last_health_check = time.time()
        
        # Wait for all processes
        while True:
            time.sleep(1)
            
            # Periodic health check
            current_time = time.time()
            if current_time - last_health_check >= health_check_interval:
                all_healthy = True
                for name, process in processes:
                    if process.poll() is not None:
                        print(f"❌ {name} process has stopped unexpectedly!")
                        all_healthy = False
                
                if all_healthy:
                    print(f"\033[36m[{time.strftime('%H:%M:%S')}] [HEALTH]\033[0m All services running normally")
                
                last_health_check = current_time
            
            # Check if any process has died
            for name, process in processes:
                if process.poll() is not None:
                    print(f"❌ {name} process has stopped!")
                    return 1
    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        
        # Terminate all processes gracefully
        for name, process in processes:
            try:
                print(f"🔄 Stopping {name}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                    print(f"✅ {name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    print(f"⚠️  Force killing {name}...")
                    process.kill()
                    process.wait()
                    print(f"✅ {name} killed")
                    
            except Exception as e:
                print(f"❌ Error stopping {name}: {e}")
        
        print("🔥 All services stopped. Goodbye!")
        return 0
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
