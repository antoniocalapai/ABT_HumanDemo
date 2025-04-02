import socket
import threading
import cv2
import pickle
import struct
import time
import numpy as np
from utils.cvfpscalc import CvFpsCalc
from queue import Queue

def frame_writer(frame_queue, client_address, frame_width, frame_height, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"{client_address[0]}.mp4", fourcc, fps, (frame_width, frame_height))
    frame_numbers_file = open(f"{client_address[0]}.txt", 'w')

    while True:
        frame, frame_number = frame_queue.get()
        if frame is None and str(frame_number) == 'END':
            break
        # Decode the JPEG encoded frame
        frame = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
        out.write(frame)
        frame_numbers_file.write(f"{frame_number}\n")
        frame_numbers_file.flush()
        print(f"{client_address[0]}: write frame {frame_number}")

    out.release()
    frame_numbers_file.close()
    print(f"{client_address[0]}: closed video writer.")
def set_socket_buffers(sock, buffer_size):
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)

def receive_data(sock):
    header_size = struct.calcsize("L")
    header = sock.recv(header_size)
    # If the number of bytes in header is less than header_size, reception failed or was incomplete
    if len(header) < header_size:
        return None
    # unpacks the header to extract the size of the data payload
    data_size = struct.unpack("L", header)[0]
    # An empty bytes object data is initialized
    data = b""
    # Receive the Data Payload
    while len(data) < data_size:
        packet = sock.recv(data_size - len(data))
        if not packet: # the connection was closed or an error occurred
            return None
        data += packet
    return data

def send_data(client_socket, data):
    try:
        serialized_data = pickle.dumps(data)
        client_socket.sendall(struct.pack("L", len(serialized_data)) + serialized_data)
        ack = client_socket.recv(1024).decode()
        return ack
    except Exception as e:
        print(e)
        return False

def jetson_handler(client_socket, client_address):

    try:
        # Receive video properties
        video_properties = receive_data(client_socket)
        video_properties = pickle.loads(video_properties)
        frame_width, frame_height, fps = video_properties
        # Send acknowledgment
        client_socket.sendall("received".encode())
        print(f"{client_address[0]}: Received video_properties: {video_properties}")

        frame_queue = Queue()
        frame_writer_thread = threading.Thread(
            target=frame_writer,
            args=(frame_queue, client_address, frame_width, frame_height, fps)
        )
        frame_writer_thread.start()

        fps_calculator = CvFpsCalc(buffer_len=10)

        while True:
            data = receive_data(client_socket)
            if data is None:
                print(f"{client_address[0]}: data is None")
                continue

            frame, frame_number = pickle.loads(data)

            # Check for "END" trigger
            if frame is None and str(frame_number) == 'END':
                frame_queue.put((None, 'END'))
                print(f"{client_address[0]}: END trigger received")
                break

            frame_queue.put((frame, frame_number))

            print(f"{client_address[0]}: received frame {frame_number}")

            # Send acknowledgment
            client_socket.sendall(str(frame_number).encode())

            # Get FPS for each video
            fps = fps_calculator.get()
            print(f"{client_address[0]}: fps {fps}")

    finally:
        frame_writer_thread.join()
        client_socket.close()
        print(f"{client_address[0]}: close client socket")

def bind_server(server_ip, server_port, buffer_size, client_num):
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

    # Bind the socket to the IP and port
    try:
        server_socket.bind((server_ip, server_port))
        # Set the buffer size for sending and receiving
        set_socket_buffers(server_socket, buffer_size)
        # Listen for incoming connections with a backlog equal to the number of Jetsons
        server_socket.listen(client_num)
        # server_socket.settimeout(100)  # Set timeout of 5 seconds
        print(f"Server listening on {server_ip}:{server_port}")
    except Exception as e:
        print(f"Server failed to bind or listen on {server_ip}:{server_port}, Error: {e}")
        server_socket.close()
        return None
    return server_socket
    
def server(server_ip, server_port, jetson_ips, buffer_size):
    
    server_socket = bind_server(server_ip, server_port, buffer_size, len(jetson_ips))

    threads = []
    connected_ips = set()

    try:
        while True:
            # Accept a connection
            client_socket, client_address = server_socket.accept()
            if str(client_address[0]) in jetson_ips:
                connected_ips.add(str(client_address[0]))
                client_handler = threading.Thread(
                    target=jetson_handler,
                    args=(client_socket, client_address)
                )
                threads.append(client_handler)
                # Break if all IPs in jetson_ips have connected
                if connected_ips == set(jetson_ips):
                    print("+++ All Jetsons have connected. +++")
                    break
            else:
                print(f"Unknown IP {client_address[0]} tried to connect.")
                client_socket.close()

        for thread in threads:
            thread.start()

    finally:
        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        # Close the server socket
        server_socket.close()


if __name__ == "__main__":
    server_ip = '192.168.1.112'  # Replace with actual server IP
    server_port = 65432  # Replace with actual server port

    # Load the IPs from the txt file
    file_path = 'ips.txt'
    credentials_dict = {'ip': [], 'user': [], 'password': []}

    with open(file_path, 'r') as file:
        header = file.readline().strip()  # Read and discard the header
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        parts = line.split('/')
        if len(parts) == 3:
            credentials_dict['ip'].append(parts[0])
            credentials_dict['user'].append(parts[1])
            credentials_dict['password'].append(parts[2])

    jetson_ips = credentials_dict['ip']
    buffer_size = 30 * 1024 * 1024 * len(jetson_ips) # 30 MB buffer size for each jetson
    server(server_ip, server_port, jetson_ips, buffer_size)