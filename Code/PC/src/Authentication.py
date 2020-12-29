import ecdsa
import socket
import threading
import time

class Authentication:
    CLIENT_PUBLIC_KEY = '98cedbb266d9fc38e41a169362708e0509e06b3040a5dfff6e08196f8d9e49cebfb4f4cb12aa7ac34b19f3b29a17f4e5464873f151fd699c2524e0b7843eb383'
    USERNAME_CLIENT = 'raffaeledndm'
    PASSWORD_CLIENT = 'ciao'

    def __init__(self, port):
        self.PORT = port
        self.sd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
    def __enter__(self):
        try:
            self.sd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sd.bind(('', self.PORT))
            self.sd.listen(5)
        except KeyboardInterrupt:
            print("Shutdown\n")
        except Exception as e:
            print(e)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.sd.close()

    def run(self):
        while(1):
            try:
                client_sd, addr = self.sd.accept()
                cl = threading.Thread(target=self.verification, args=(client_sd, addr))
                cl.start()
            except KeyboardInterrupt:
                print("Shutdown\n")
                return
            except Exception as e:
                print(e)

    def verification(self, client_sd, addr):
        time.sleep(4)
        #vk = VerifyingKey.from_string(bytes.fromhex(), curve=ecdsa.SECP256k1)
        #vk.verify(bytes.fromhex(sig), message) # True


with Authentication(8080) as a:
    a.run()