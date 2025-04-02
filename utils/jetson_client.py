import socket
import cv2
import pickle
import struct
import time
from utils import CvFpsCalc

def set_socket_buffers(sock, buffer_size): # Adjusting Buffer Size
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)

def send_data(sock, data):
    serialized_data = pickle.dumps(data)
    sock.sendall(struct.pack("L", len(serialized_data)) + serialized_data)

def connect_to_server(server_ip, server_port, retry_interval, buffer_size):
    while True:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Set the buffer size for sending and receiving
            set_socket_buffers(client_socket, buffer_size)
            client_socket.connect((server_ip, server_port))
            print(f"Connected to server {server_ip}.")
            return client_socket
        except ConnectionRefusedError:
            print(f"Connecting to {server_ip} server refused. Retrying...")
            time.sleep(retry_interval)
    print(f"Failed to connect to server {server_ip}.")
    return None

def main():
    server_ip = '192.168.1.3'  # Replace with the computer's IP address
    server_port = 65432
    buffer_size = 30 * 1024 * 1024  # 30 MB buffer size

    video_path = 'koala4k.mp4' # load a video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_properties = (frame_width, frame_height, fps)

    client_socket = connect_to_server(server_ip, server_port, 1, buffer_size)
    if client_socket is None:
        print(f"connecting to server {server_ip} failed. Exiting...")
        return

    try:
        # Send video properties
        while True:
            send_data(client_socket, video_properties)
            # Get acknowledgement
            ack = client_socket.recv(1024).decode()
            if ack == "received":
                print(f"video_properties sent to {server_ip} successfully.")
                break
            else:
                continue

        frame_number = 1
        new_frame = True
        fps_calculator = CvFpsCalc(buffer_len=10)
        while True:
            if new_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                # Encode frame as JPEG image
                _, encoded_frame = cv2.imencode('.jpg', frame)
                frame_data = encoded_frame.tobytes()

            send_data(client_socket, (frame_data, frame_number))

            # Get acknowledgement
            ack = client_socket.recv(1024).decode()
            if ack == str(frame_number):
                print(f"Frame {frame_number} sent successfully.")
                frame_number += 1
                new_frame = True
            else:
                new_frame = False
                continue

            # Get FPS
            fps = fps_calculator.get()
            print(f"fps: {fps}")

        # Send end-of-video trigger
        send_data(client_socket, (None, 'END'))
    finally:
        cap.release()
        client_socket.close()
        print("Exit.")


if __name__ == '__main__':
    main()
