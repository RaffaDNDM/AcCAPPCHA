import ecdsa
import socket
import threading
import time
import hashlib
from termcolor import cprint
import ssl

class InvalidMessage(Exception):
    pass

class Authentication:
    USERNAME_CLIENT = 'raffaeledndm'
    PASSWORD_CLIENT = 'ciao'
    NONCE_LENGTH = 16
    SIGNATURE_LENGTH = 64
    USERS = {}

    def __init__(self, port):
        with open('../dat/crypto/ecdsa.pem', "r") as sk_file:
            sk_pem = sk_file.read().encode()
            self.ECDSA_CLIENT_PUBLIC_KEY = ecdsa.VerifyingKey.from_pem(sk_pem)

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

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.sd.close()
        print('d')

    def run(self):
        while(True):
            try:
                client_sd, addr = self.sd.accept()
                client_sd = ssl.wrap_socket(client_sd, server_side=True, 
                                              ca_certs = "../dat/crypto/client.pem",
                                              certfile="../dat/crypto/server.pem",
                                              keyfile="../dat/crypto/server.key",
                                              cert_reqs=ssl.CERT_REQUIRED,
                                              ssl_version=ssl.PROTOCOL_TLSv1_2)

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
        print(addr)

        try:
            #hash_msg = hashlib.sha256(msg.encode()+nonce).hexdigest()
            self.ECDSA_CLIENT_PUBLIC_KEY.verify(signature, msg.encode()+nonce)

            check = True           
            user = f'{addr[0]}:{addr[1]}'
            if user in list(self.USERS.keys()):
                if nonce in self.USERS[user]:
                    #Replay attack
                    check = False
                else:
                    #Authorized (Tracking all the nonce used in this session)
                    self.USERS[user].append(nonce)    
            else:
                #Authorized
                self.USERS[user]=[]


            if check:
                if msg == 'True':
                    client_sd.send(b'OK\r\n')
                elif msg == 'False':
                    client_sd.send(b'NO\r\n')
                else:
                    client_sd.send(b'ERROR\r\n')
            else:
                client_sd.send(b'NO\r\n')
                

        except ecdsa.keys.BadSignatureError:
            # NO correspondence between signature and sign(msg+nonce)
            client_sd.send(b'Unauthorized\r\n')
        
        client_sd.send('OK\r\n'.encode())
        #vk = VerifyingKey.from_string(bytes.fromhex(), curve=ecdsa.SECP256k1)
        #vk.verify(bytes.fromhex(sig), message) # True


with Authentication(8080) as a:
    a.run()