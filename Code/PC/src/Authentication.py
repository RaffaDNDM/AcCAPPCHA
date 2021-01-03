import ecdsa
import socket
import threading
import time
import hashlib
from termcolor import cprint
import ssl
import psycopg2

class InvalidMessage(Exception):
    pass

class Authentication:
    USERNAME_CLIENT = 'raffaeledndm'
    PASSWORD_CLIENT = 'ciao'
    NONCE_LENGTH = 16
    SIGNATURE_LENGTH = 64
    USERS = {}
    FOLDER_HTML = '../dat/html/'
    LOGGED_FILE = 'logged.html'
    FAILURE_FILE = 'failure.html'
    NO_DB_ENTRY_FILE = 'no_db_entry.html'

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
            user = addr[0]
            if user in list(self.USERS.keys()):
                if nonce in self.USERS[user]:
                    #Replay attack
                    check = False
                else:
                    #Authorized (Tracking all the nonce used in this session)
                    self.USERS[user].append(nonce)    
            else:
                #Authorized
                self.USERS[user]=[nonce,]


            if check:
                if msg == 'True':
                    client_sd.send('OK\r\n'.encode())
                    self.authentication()
                elif msg == 'False':
                    client_sd.send('NO\r\n'.encode())
                else:
                    client_sd.send('ERROR\r\n'.encode())
            else:
                client_sd.send('NO\r\n'.encode())
                
        except ecdsa.keys.BadSignatureError:
            # NO correspondence between signature and sign(msg+nonce)
            client_sd.send(b'Unauthorized\r\n')
        
        #vk = VerifyingKey.from_string(bytes.fromhex(), curve=ecdsa.SECP256k1)
        #vk.verify(bytes.fromhex(sig), message) # True

    def authentication(self):
        request_header = ''

        while(True):
            char = self.sd.recv(1).decode('utf-8', 'ignore')
            request_header += char

            if request_header.endswith('\r\n\r\n'):
                break

        request_header = request_header[:-4]
        header_list = request_header.split('\r\n')
        request_line = header_list[0].split(' ')
        headers = {x.split(': ', 1)[0]:x.split(': ', 1)[1] for x in header_list[1:]}

        if request_line == ['POST', '/auth', 'HTTP/1.1']:
            length_body = int(headers['Content-Length'])
            body = self.sd.recv(length_body).decode('utf-8', 'ignore')
            parameter_list = body.split('&')
            
            if len(parameter_list) != 2:
                self.sd.send(b'HTTP/1.1 400 Bad Request\r\n\r\n')

            parameters = {x.split('=')[0]:x.split('=')[1] for x in parameter_list}
            self.auth(parameters)

        elif request_line[1]!='/auth':
            self.sd.send(b'HTTP/1.1 501 Not Implemented\r\n\r\n')

    def auth(self, parameters):
        self.sd.send(b'HTTP/1.1 200 OK\r\n')
        file_path = ''

        if parameters['user']==self.USERNAME_CLIENT:
            if parameters['pwd']==self.PASSWORD_CLIENT:
                file_path = self.FOLDER_HTML + self.LOGGED_FILE
            else:
                file_path = self.FOLDER_HTML + self.FAILURE_FILE
        else:
            file_path = self.FOLDER_HTML + self.NO_DB_ENTRY_FILE

        with open(file_path, 'r') as f:
            body = f.read()

        msg = f'Content-Length: {len(body)}\r\n\r\n'+ \
               body

        self.sd.send(msg.encode())


with Authentication(8080) as a:
    a.run()