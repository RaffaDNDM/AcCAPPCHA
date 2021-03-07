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
    SIZE_SIGN_REQUEST = 100
    SIZE_SIGN_RESPONSE = 10
    SIZE_POST_REQUEST = 300
    SIZE_POST_RESPONSE = 300
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

        SIZE_SIGN_REQUEST (int): Length (#bytes) of the message (m, n, sign(m||n))
                                 sent by the client
        
        SIZE_SIGN_RESPONSE (int): Length (#bytes) of the message, sent by the
                                  server, to answer to (m, n, sign(m||n))
    
        SIZE_POST_REQUEST (int): Length (#bytes) of the POST request, containing
                                 the user's credentials, sent by the client
        
        SIZE_POST_RESPONSE (int): Length (#bytes) of the message, sent by the
                                  server, to answer to POST request
        
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
        #Read public key of the client
        with open('../dat/crypto/ecdsa.pem', "r") as sk_file:
            sk_pem = sk_file.read().encode()
            self.ECDSA_CLIENT_PUBLIC_KEY = ecdsa.VerifyingKey.from_pem(sk_pem)

        #Creation of the TCP socket
        self.sd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #Server information
        self.PORT = port
        #Debug information
        self.DEBUG = debug_option
        
    def __enter__(self):
        """
        Bind and listen for client request
        """

        try:
            #Reusability option
            self.sd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            #Server works on 127.0.0.1 on port specified in the constructor
            self.sd.bind(('', self.PORT))
            #Server has a queue of maximum 5 pending requests
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
                #Receive a client request of connection
                client_sd, addr = self.sd.accept()
                #Wrap TCP socket of the client in TLS socket
                client_sd = ssl.wrap_socket(client_sd, server_side=True, 
                                              ca_certs = "../dat/crypto/client.pem",
                                              certfile="../dat/crypto/server.pem",
                                              keyfile="../dat/crypto/server.key",
                                              cert_reqs=ssl.CERT_REQUIRED,
                                              ssl_version=ssl.PROTOCOL_TLSv1_2)
                
                #Manage the client with a thread
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
        
        #Response of the verification (m, n, sign(m||n))
        msg = ''

        while(True):
            char = client_sd.recv(1).decode('utf-8', 'ignore')
            msg += char

            if msg.endswith('\r\n'):
                break

        msg = msg[:-2]

        if len(msg.split('\r\n'))>3:
            raise InvalidMessage

        nonce = client_sd.recv(self.NONCE_LENGTH)
        signature = client_sd.recv(self.SIGNATURE_LENGTH)
        n = len(msg)+2+len(nonce)+len(signature)
        padding = client_sd.recv(self.SIZE_SIGN_REQUEST-n)
        cprint(f'{addr}', 'blue')

        #Debug information
        if self.DEBUG:
            cprint(self.LINE, 'blue')
            cprint(msg, 'cyan')
            cprint(nonce, 'blue')
            cprint(signature, 'green')
        
        try:
            #ECDSA verification
            self.ECDSA_CLIENT_PUBLIC_KEY.verify(signature, msg.encode()+nonce)
            check = True           
            user = addr[0]

            if user in list(self.USERS.keys()):
                if nonce in self.USERS[user]:
                    #Replay attack
                    check = False
                else:
                    #Authorized (not the first request of the user)
                    self.USERS[user].append(nonce)    
            else:
                #Authorized (first request of the user)
                self.USERS[user]=[nonce,]

            if check:
                #Authorized user (ECDSA ok, Nonce ok)

                if msg == 'True':
                    #The user was a human
                    response = self.pad_msg(b'OK\r\n', self.SIZE_SIGN_RESPONSE)
                    client_sd.send(response)
                    self.authentication(client_sd)
                
                elif msg == 'False':
                    #The user was a bot
                    response = self.pad_msg(b'NO\r\n', self.SIZE_SIGN_RESPONSE)
                    client_sd.send(response)
                
                else:
                    #The response of the user's evaluation has invalid format
                    response = self.pad_msg(b'ERROR\r\n', self.SIZE_SIGN_RESPONSE)
                    client_sd.send(response)            
            else:
                #Unauthorized user (ECDSA ok, Nonce no)
                response = self.pad_msg(b'NO\r\n', self.SIZE_SIGN_RESPONSE)
                client_sd.send(response)
                
        except ecdsa.keys.BadSignatureError:
            # No integrity in ECDSA [(msg+nonce)!=sign^(-1)(signature)]
            response = self.pad_msg(b'No integrity\r\n', self.SIZE_SIGN_RESPONSE)
            client_sd.send(response)
        
        #Close the client socket
        client_sd.close()

    def authentication(self, client_sd):
        """
        Wait for POST request of user and manage it
        replying to him

        Args:
            client_sd (ssl.SSLSocket): SSL socket for communication 
                                       with client
        """

        #POST request from the client
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
            #Analysis of the parameters in the HTTP POST request
            length_body = int(headers['Content-Length'])
            body = client_sd.recv(length_body).decode('utf-8', 'ignore')
            parameter_list = body.split('&')
            n = len(request_header)+length_body
            padding = client_sd.recv(self.SIZE_POST_REQUEST-n)
            
            if len(parameter_list) != 2:
                #Number of parameters != 2 (username, password)
                response = self.pad_msg(b'HTTP/1.1 400 Bad Request\r\n\r\n', self.SIZE_POST_RESPONSE)
                client_sd.send(response)

            #Dictionary of parameters (name_parameter, value)
            parameters = {x.split('=')[0]:x.split('=')[1] for x in parameter_list}
            #Authenticate the user
            self.auth(client_sd, parameters)

        elif request_line[1]!='/cgi-bin/auth' or request_line[0]!='POST':
            #Invalid request by the client
            response = self.pad_msg(b'HTTP/1.1 501 Not Implemented\r\n\r\n', self.SIZE_POST_RESPONSE)
            client_sd.send(response)

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
        
        #Response to the client for the POST request
        status_line = b'HTTP/1.1 200 OK\r\n'
        file_path = ''

        #Search of credentials in the database
        try:
            #Connection to the database
            connection = psycopg2.connect(user="postgres",
                                    password="postgres",
                                    host="127.0.0.1",
                                    port="5432",
                                    database="cloudservice")

            cursor = connection.cursor()

            #Search the hashed password of the specified Username=parameters['user'] 
            cursor.execute(f"SELECT Password FROM CloudUser WHERE Username = '{parameters['user']}'")
            hash_pssword = cursor.fetchone()

            if hash_pssword is None:
                #No entry in DB with Username = parameters['user']
                file_path = self.FOLDER_HTML + self.NO_DB_ENTRY_FILE
            else:
                #Compare password in the POST
                if parameters['pwd']==hash_pssword[0]:
                    #Correct password
                    file_path = self.FOLDER_HTML + self.LOGGED_FILE
                else:
                    #Wrong password
                    file_path = self.FOLDER_HTML + self.FAILURE_FILE

            #Read the file related to the status of the search
            with open(file_path, 'r') as f:
                body = f.read()

            #Send response to the client
            response = status_line+ (f'Content-Length: {len(body)}\r\n\r\n'+ body).encode()
            response = self.pad_msg(response, self.SIZE_POST_RESPONSE)
            client_sd.send(response)

        except (Exception, Error) as error:
            print("Error while connecting to PostgreSQL", error)
        
        finally:
            #Close the connection with the database
            if (connection):
                cursor.close()
                connection.close()

                if self.DEBUG:
                    print("PostgreSQL connection is closed")
                    cprint(self.LINE, 'blue', end='\n\n')

    def pad_msg(self, msg, size):
        """
        Pad a message to a fixed length message using space characters
        
        Args:
            msg (bytes): Bytes message to be padded

            size (int): Size of final padded message

        Returns:
            padded_msg (bytes): Padded message of size bytes
        """

        if len(msg)<size:
            #Padding of the message
            msg = msg + b' '*(size-len(msg))

        elif len(msg)<size:
            #Message longer than size bytes
            print(f'Padding error, msg size is bigger than maximum ({size})')

        return msg     

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