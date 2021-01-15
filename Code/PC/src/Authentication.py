import ecdsa
import socket
import threading
import time
import hashlib
import argparse
from termcolor import cprint
import ssl
import psycopg2
from psycopg2 import Error

class InvalidMessage(Exception):
    pass

class Authentication:
    NONCE_LENGTH = 16
    SIGNATURE_LENGTH = 64
    USERS = {}
    FOLDER_HTML = '../dat/html/'
    LOGGED_FILE = 'logged.html'
    FAILURE_FILE = 'failure.html'
    NO_DB_ENTRY_FILE = 'no_db_entry.html'
    LINE = '_____________________________________________________'

    """
    Authentication object performs signature verification
    and the authentication phase of the remote client

    Args:
        port (int): Port number on which the server works

        debug_option (bool): True if you want to show more debugging info
                             during the execution, False otherwise

    Attributes:
        PORT (int):  Port number on which the server works

        DEBUG (bool): True if you want to show more debugging info
                      during the execution, False otherwise

        sd (socket.socket): TCP socket instance on which server will work

        ECDSA_CLIENT_PUBLIC_KEY (ecdsa.VerifyingKey): client public key for 
                                                      ECDSA verification

        NONCE_LENGTH (int): Length (#bytes) of nonce in message (uuid4 nonce)
        
        SIGNATURE_LENGTH (int): Length (#bytes) of signature of msg + nonce
        
        USERS (dict): Dictionary of elements composed by pairs (key, value):
                      key (str): IP address
                      value (list): list of nonce used by 'key' IP address
        
        FOLDER_HTML (str): Folder that contains HTML files for response to
                           authentication
        
        LOGGED_FILE (str): Name of HTML file for success in authentication
        
        FAILURE_FILE (str): Name of HTML file for wrong password in 
                            authentication
        
        NO_DB_ENTRY_FILE (str): Name of HTML file for not existing user with
                                the username, specified by client, in DB
    """
    def __init__(self, port, debug_option):
        with open('../dat/crypto/ecdsa.pem', "r") as sk_file:
            sk_pem = sk_file.read().encode()
            self.ECDSA_CLIENT_PUBLIC_KEY = ecdsa.VerifyingKey.from_pem(sk_pem)

        self.PORT = port
        self.DEBUG = debug_option
        self.sd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def __enter__(self):
        """
        Bind and listen for client request
        """

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
        """
        Close the socket stream
        """
        
        self.sd.close()

    def run(self):
        """
        Manage each client request
        """

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
        """
        Veify (nonce, message, signature) from client
        using ECDSA signature

        Args:
            client_sd (ssl.SSLSocket): SSL socket for communication 
                                       with client

            addr (tuple): Tuple composed by (IP_address, port) where
                          IP_address (str): IP address of the client
                          port (int): Port number of the client
        """
        
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
        cprint(f'{addr}', 'blue')

        if self.DEBUG:
            cprint(self.LINE, 'blue')
            cprint(msg, 'cyan')
            cprint(nonce, 'blue')
            cprint(signature, 'green')
        
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
                    self.authentication(client_sd)
                elif msg == 'False':
                    client_sd.send('NO\r\n'.encode())
                else:
                    client_sd.send('ERROR\r\n'.encode())
            else:
                client_sd.send('NO\r\n'.encode())
                
        except ecdsa.keys.BadSignatureError:
            # NO correspondence between signature and sign(msg+nonce)
            client_sd.send(b'No integrity\r\n')
        
        #vk = VerifyingKey.from_string(bytes.fromhex(), curve=ecdsa.SECP256k1)
        #vk.verify(bytes.fromhex(sig), message) # True
        client_sd.close()

    def authentication(self, client_sd):
        """
        Wait for POST request of user and manage it
        replying to him

        Args:
            client_sd (ssl.SSLSocket): SSL socket for communication 
                                       with client
        """
        request_header = ''

        while(True):
            char = client_sd.recv(1).decode('utf-8', 'ignore')
            request_header += char

            if request_header.endswith('\r\n\r\n'):
                break

        request_header = request_header[:-4]
        header_list = request_header.split('\r\n')
        request_line = header_list[0].split(' ')
        headers = {x.split(': ', 1)[0]:x.split(': ', 1)[1] for x in header_list[1:]}

        if request_line == ['POST', '/cgi-bin/auth', 'HTTP/1.1']:
            length_body = int(headers['Content-Length'])
            body = client_sd.recv(length_body).decode('utf-8', 'ignore')
            parameter_list = body.split('&')
            
            if len(parameter_list) != 2:
                client_sd.send(b'HTTP/1.1 400 Bad Request\r\n\r\n')

            parameters = {x.split('=')[0]:x.split('=')[1] for x in parameter_list}
            self.auth(client_sd, parameters)

        elif request_line[1]!='/cgi-bin/auth':
            client_sd.send(b'HTTP/1.1 501 Not Implemented\r\n\r\n')

    def auth(self, client_sd, parameters):
        """
        Check if the password inserted by the user was correct
        looking to DB and reply with HTML page to client

        Args:
            client_sd (ssl.SSLSocket): SSL socket for communication 
                                       with client

            parameters (dict): Dictionary composed by two (key, value) pairs.
                               The two entries are: 
                               ('user', value) where value is the username
                               ('pwd', value) where value is his password
        """
        
        client_sd.send(b'HTTP/1.1 200 OK\r\n')
        file_path = ''

        try:
            connection = psycopg2.connect(user="postgres",
                                    password="postgres",
                                    host="127.0.0.1",
                                    port="5432",
                                    database="cloudservice")

            cursor = connection.cursor()

            # Read PostgreSQL purchase timestamp value into Python datetime
            cursor.execute(f"SELECT Password FROM CloudUser WHERE Username = '{parameters['user']}'")
            hash_pssword = cursor.fetchone()

            if hash_pssword is None:
                #No entry in DB with Username = parameters['user']
                file_path = self.FOLDER_HTML + self.NO_DB_ENTRY_FILE
            else:
                if parameters['pwd']==hash_pssword[0]:
                    file_path = self.FOLDER_HTML + self.LOGGED_FILE
                else:
                    file_path = self.FOLDER_HTML + self.FAILURE_FILE

            with open(file_path, 'r') as f:
                body = f.read()

            msg = f'Content-Length: {len(body)}\r\n\r\n'+ \
                body

            client_sd.send(msg.encode())

        except (Exception, Error) as error:
            print("Error while connecting to PostgreSQL", error)
        
        finally:
            if (connection):
                cursor.close()
                connection.close()

                if self.DEBUG:
                    print("PostgreSQL connection is closed")
                    cprint(self.LINE, 'blue', end='\n\n')

def args_parser():
    '''
    Parser of command line arguments
    '''
    #Parser of command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-debug", "-dbg", 
                        dest="debug",
                        help="""If specified, it shows debug info""",
                        action='store_true')

    #Parse command line arguments
    args = parser.parse_args()

    return args.debug

def main():
    debug_option = args_parser()

    with Authentication(8080, debug_option) as a:
        a.run()

if __name__=='__main__':
    main()