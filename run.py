"""
Main entry point for Federated Learning system
Coordinates server and client startup
"""
import os
import sys
import subprocess
import time
from dotenv import load_dotenv
import multiprocessing
from sklearn.model_selection import train_test_split

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.data_utils import (
    download_liver_dataset, preprocess_data, 
    create_federated_splits, save_test_data
)
import pickle

load_dotenv()


def prepare_data():
    """Download and prepare data for federated learning"""
    print("=" * 60)
    print("📊 STEP 1: DATA PREPARATION")
    print("=" * 60)
    
    # Download dataset
    X, y, feature_names = download_liver_dataset()
    
    # Preprocess
    print("\n🔄 Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # Save test data
    save_test_data(X_test, y_test)
    
    # Create federated splits
    num_clients = int(os.getenv('NUM_CLIENTS', 3))
    print(f"\n🏥 Creating federated splits for {num_clients} clients...")
    client_data = create_federated_splits(X_train, y_train, num_clients)
    
    # Save client data
    os.makedirs('data', exist_ok=True)
    for i, data in enumerate(client_data):
        # Split each client's data into train and validation
        X_client_train, X_client_val, y_client_train, y_client_val = train_test_split(
            data['X'], data['y'], test_size=0.2, random_state=42
        )
        
        client_file = f'data/client_{i}_data.pkl'
        with open(client_file, 'wb') as f:
            pickle.dump({
                'X_train': X_client_train,
                'y_train': y_client_train,
                'X_val': X_client_val,
                'y_val': y_client_val
            }, f)
        print(f"   💾 Saved {client_file}")
    
    print("\n✅ Data preparation complete!\n")
    return num_clients


def start_flower_server():
    """Start Flower federated learning server"""
    print("=" * 60)
    print("🌍 STEP 2: STARTING FLOWER SERVER")
    print("=" * 60)
    
    server_process = subprocess.Popen(
        [sys.executable, 'backend/federated_server.py'],
        cwd=os.getcwd()
    )
    
    print("✅ Flower server started!\n")
    return server_process


def start_flower_clients(num_clients):
    """Start Flower clients"""
    print("=" * 60)
    print(f"🏥 STEP 3: STARTING {num_clients} FLOWER CLIENTS")
    print("=" * 60)
    
    # Wait for server to be ready
    time.sleep(5)
    
    client_processes = []
    for i in range(num_clients):
        print(f"   Starting Client {i}...")
        process = subprocess.Popen(
            [sys.executable, 'backend/federated_client.py', str(i)],
            cwd=os.getcwd()
        )
        client_processes.append(process)
        time.sleep(1)  # Stagger client starts
    
    print("\n✅ All clients started!\n")
    return client_processes


def start_api_server():
    """Start Flask API server"""
    print("=" * 60)
    print("🌐 STEP 4: STARTING WEB INTERFACE")
    print("=" * 60)
    
    port = int(os.getenv('FLASK_PORT', 5000))
    print(f"   Starting Flask server on port {port}...")
    
    api_process = subprocess.Popen(
        [sys.executable, 'backend/api_server.py'],
        cwd=os.getcwd()
    )
    
    print(f"\n✅ Web interface running at http://localhost:{port}")
    print("\n" + "=" * 60)
    print("🎉 FEDERATED LEARNING SYSTEM READY!")
    print("=" * 60)
    print(f"\n📱 Open your browser to: http://localhost:{port}")
    print("🔄 Federated training will complete shortly...")
    print("\n⚠️  Press Ctrl+C to stop all services\n")
    
    return api_process


def cleanup_processes(processes):
    """Clean up all processes"""
    print("\n\n🛑 Shutting down services...")
    for process in processes:
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
    print("✅ Cleanup complete!")


def main():
    """Main execution"""
    processes = []
    
    try:
        # Step 1: Prepare data
        num_clients = prepare_data()
        
        # Step 2: Start Flower server
        server_process = start_flower_server()
        processes.append(server_process)
        
        # Step 3: Start clients
        client_processes = start_flower_clients(num_clients)
        processes.extend(client_processes)
        
        # Wait for federated training to complete
        # The server will exit automatically after all rounds
        server_process.wait()
        
        print("\n" + "=" * 60)
        print("✅ FEDERATED TRAINING COMPLETE!")
        print("=" * 60)
        
        # Cleanup client processes
        for client in client_processes:
            try:
                client.terminate()
                client.wait(timeout=3)
            except:
                client.kill()
        
        # Step 4: Start API server for web interface
        api_process = start_api_server()
        processes = [api_process]
        
        # Keep running until interrupted
        api_process.wait()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Received interrupt signal...")
        cleanup_processes(processes)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        cleanup_processes(processes)
        sys.exit(1)


if __name__ == '__main__':
    main()
