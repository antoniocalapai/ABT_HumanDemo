import paramiko


def is_jetson_on(ip, user, password):
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(ip, username=user, password=password)
        client.close()
        return True
    except Exception as e:
        return False


def is_gui_running(ip, user, password, process_name):
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(ip, username=user, password=password)

        command = f"pgrep -f {process_name}"
        stdin, stdout, stderr = client.exec_command(command)
        process_ids = stdout.read().decode().strip()

        client.close()

        if process_ids:
            return True
        else:
            return False
    except Exception as e:
        print(f"Failed to check process on Jetson: {e}")
        return False
