import serial
import serial.tools.list_ports
import time
import csv
import os
import datetime
import sys

# Configuration
BAUD_RATE = 115200
DATA_DIR = os.path.join("data", "raw")
CSV_HEADER = ["t_ms", "ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz"]

def list_serial_ports():
    """Lists available serial ports."""
    ports = serial.tools.list_ports.comports()
    return ports

def select_port():
    """Lists ports and asks user to select one."""
    ports = list_serial_ports()
    if not ports:
        print("No serial ports found. Please check your connection.")
        return None
    
    print("\nAvailable Serial Ports:")
    for i, port in enumerate(ports):
        print(f"{i+1}: {port.device} - {port.description}")
    
    while True:
        try:
            selection = input("\nSelect port number (or 'q' to quit): ")
            if selection.lower() == 'q':
                return None
            index = int(selection) - 1
            if 0 <= index < len(ports):
                return ports[index].device
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def parse_line(line):
    """
    Parses a line of text from the sensor.
    Returns a list of values if valid data, None otherwise.
    Expected format: t_ms,ax,ay,az,gx,gy,gz,mx,my,mz
    """
    try:
        line = line.strip()
        if not line:
            return None
        
        # Split by comma
        parts = line.split(',')
        
        # Check if we have the correct number of fields (10)
        if len(parts) != 10:
            return None
        
        # Try converting to floats to ensure it's data
        # t_ms is technically int but float is fine for storage
        values = [float(x) for x in parts]
        return values
    except ValueError:
        # Not a data line (likely debug text or header)
        return None

def main():
    print("--- M5Stack Sensor Data Capture ---")
    
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Select port
    port = select_port()
    if not port:
        print("Exiting.")
        return

    print(f"Connecting to {port} at {BAUD_RATE} baud...")
    
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        
        # DTR/RTS toggle to wake up/reset some ESP32/M5Stack boards
        ser.dtr = False
        ser.rts = False
        time.sleep(0.1)
        ser.dtr = True
        ser.rts = True
        
        time.sleep(2) # Wait for connection to stabilize
        print("Connected!")
        
        # Create filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sensor_data_{timestamp}.csv"
        filepath = os.path.join(DATA_DIR, filename)
        
        print(f"Saving data to: {filepath}")
        print("Press Ctrl+C to stop capturing.")
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(CSV_HEADER)
            
            line_count = 0
            start_time = time.time()
            
            # Flush input buffer to remove old data
            ser.reset_input_buffer()
            
            while True:
                if ser.in_waiting > 0:
                    try:
                        # Read line from serial
                        line_bytes = ser.readline()
                        line_str = line_bytes.decode('utf-8', errors='replace').strip()
                        
                        # Parse data
                        data = parse_line(line_str)
                        
                        if data:
                            writer.writerow(data)
                            line_count += 1
                            
                            # Provide feedback every 50 lines (approx 1 sec at 50Hz)
                            if line_count % 50 == 0:
                                elapsed = time.time() - start_time
                                rate = line_count / elapsed if elapsed > 0 else 0
                                sys.stdout.write(f"\rCaptured {line_count} samples. Rate: {rate:.1f} Hz. Last: {data[0]:.0f}ms")
                                sys.stdout.flush()
                        else:
                            # Print non-data lines for debugging
                            if line_str:
                                print(f"RAW: {line_str}")
                                
                    except UnicodeDecodeError:
                        pass # Ignore decoding errors
                        
    except serial.SerialException as e:
        print(f"\nError opening serial port: {e}")
    except KeyboardInterrupt:
        print(f"\n\nCapture stopped by user.")
        print(f"Total samples saved: {line_count}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial connection closed.")

if __name__ == "__main__":
    main()

