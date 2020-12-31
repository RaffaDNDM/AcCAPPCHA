import ecdsa
import socket
import threading
import time
import hashlib
from termcolor import cprint

class InvalidMessage(Exception):
    pass

class Authentication:
    CLIENT_PUBLIC_KEY = '9aedaaa1468e178426e7cfb7257bee4ee589106ded1f282bc2cde90ebaea07ca'+ \
                        '83ff329740d42624069f77fa360a6c30d2fc26ed7ce9cc32ed97cd5865aaa3c6'
    USERNAME_CLIENT = 'raffaeledndm'
    PASSWORD_CLIENT = 'ciao'
    NONCE_LENGTH = 16
    SIGNATURE_LENGTH = 64

    def __init__(self, port):
        self.PORT = port
        self.sd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('a')

    def __enter__(self):
        try:
            self.sd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sd.bind(('', self.PORT))
            self.sd.listen(5)
            print('b')
        except KeyboardInterrupt:
            print("Shutdown\n")
        except Exception as e:
            print(e)

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.sd.close()
        print('d')

    def run(self):
        while(True):
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
        #Decoded Message
        msg = ''

        while(True):
            char = client_sd.recv(1).decode('utf-8', 'ignore')
            msg += char

            if msg.endswith('\r\n'):
                break

        msg = msg[:-2]

        if len(msg.split('\r\n'))>3:
            raise InvalidMessage

        #Encoded nonce and signature
        nonce = client_sd.recv(self.NONCE_LENGTH)
        signature = client_sd.recv(self.SIGNATURE_LENGTH)
        cprint(msg, 'cyan')
        cprint(nonce, 'blue')
        cprint(signature, 'green')
        
        try:
            verify_key = ecdsa.VerifyingKey.from_string(bytes.fromhex(self.CLIENT_PUBLIC_KEY), curve=ecdsa.SECP256k1)
            hash_msg = hashlib.sha256(msg.encode()+nonce).hexdigest()
            print(verify_key.verify(signature, hash_msg.encode()))

        except ecdsa.keys.BadSignatureError:
            return False
        
        client_sd.send('OK\r\n'.encode())
        #vk = VerifyingKey.from_string(bytes.fromhex(), curve=ecdsa.SECP256k1)
        #vk.verify(bytes.fromhex(sig), message) # True


with Authentication(8080) as a:
    a.run()