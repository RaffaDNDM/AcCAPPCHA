import ecdsa
import socket
import uuid
from termcolor import cprint, colored
import hashlib
import ssl

class SecureElement:
    # SECP256k1 is the Bitcoin elliptic curve
    #ECDSA_PRIVATE_KEY = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1, hashfunc=hashlib.sha256)
    #with open('../dat/crypto/ecdsa.key', 'w') as private_pem:
    #    private_pem.write(ECDSA_PRIVATE_KEY.to_pem().decode())

    #PUBLIC_KEY = ECDSA_PRIVATE_KEY.get_verifying_key()
    #with open('../dat/crypto/ecdsa.pem', 'w') as public_pem:
    #    public_pem.write(PUBLIC_KEY.to_pem().decode())

    """
    Authentication object performs signature verification
    and the authentication phase of the remote client

    Args:
        IP_address (str): IP address of the authentication server
        
        port (int): Port number of the authentication server

    Attributes:
        IP_ADDRESS (str): IP address of the authentication server
        
        PORT (int):  Port number on which the server works

        sd (socket.socket): TCP socket instance on which client will work

        ECDSA_PRIVATE_KEY (ecdsa.SigningKey): private key for ECDSA signing

        SIZE_SIGN_REQUEST (int): Length (#bytes) of the message (m, n, sign(m||n))
                                 sent by the client
        
        SIZE_SIGN_RESPONSE (int): Length (#bytes) of the message, sent by the
                                  server, to answer to (m, n, sign(m||n))
    
        SIZE_POST_REQUEST (int): Length (#bytes) of the POST request, containing
                                 the user's credentials, sent by the client
        
        SIZE_POST_RESPONSE (int): Length (#bytes) of the message, sent by the
                                  server, to answer to POST request

    """
    SIZE_SIGN_REQUEST = 100
    SIZE_SIGN_RESPONSE = 10
    SIZE_POST_REQUEST = 300
    SIZE_POST_RESPONSE = 300

    def __init__(self, IP_address, port):
        #Read its private key
        with open('../dat/crypto/ecdsa.key', "r") as sk_file:
            sk_pem = sk_file.read().encode()
            self.ECDSA_PRIVATE_KEY = ecdsa.SigningKey.from_pem(sk_pem)

        #Creation of TCP socket
        self.sd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #Server information
        self.IP_ADDRESS = IP_address
        self.PORT = port
        
    def __enter__(self):
        """
        Connect to server after wrapping socket in SSL socket
        """
        
        #Wrap TCP socket in TLS socket
        self.sd = ssl.wrap_socket(self.sd,
                                  server_side=False,
                                  ca_certs = "../dat/crypto/server.pem", 
                                  certfile="../dat/crypto/client.pem",
                                  keyfile="../dat/crypto/client.key",
                                  cert_reqs=ssl.CERT_REQUIRED,
                                  ssl_version=ssl.PROTOCOL_TLSv1_2)

        #Connection to the server (check certificate and keys)
        self.sd.connect((self.IP_ADDRESS, self.PORT))

        return self

    def sign(self, msg):
        """
        Sign a message, send it to the server and wait for
        the response

        Args:
            msg (str): Message to be signed with a nonce 

        Returns:
            response (bool): True if human, False otherwise
        """
        
        #Send verification response (m, n, sign(m||n))
        nonce=uuid.uuid4().bytes
        signature = self.ECDSA_PRIVATE_KEY.sign(msg.encode()+nonce)
        result = msg.encode()+b'\r\n'+nonce+signature        
        result = self.pad_msg(result, self.SIZE_SIGN_REQUEST)        
        self.sd.send(result)

        #Wait for AcCAPPCHA response
        check = ''

        while(True):
            char = self.sd.recv(1).decode('utf-8', 'ignore')
            check += char

            if check.endswith('\r\n'):
                break

        n = len(check)
        padding = self.sd.recv(self.SIZE_SIGN_RESPONSE-n)
            
        #Check the response
        if check[:-2] == 'OK':
            #Human
            return True
        else:
            #Bot or error
            return False

    def credentials(self, username, password):
        """
        Send HTTP POST request with credentials and wait for
        response HTML page

        Args:
            username (str): Username inserted by user

            password (str): Password to be hashed with SHA512

        Returns:
            response (bool): True if human, False otherwise
        """

        #Send credentials using a POST request
        hash_pwd = hashlib.sha512(password.encode()).hexdigest()
        body = f'user={username}&pwd={hash_pwd}'

        request = 'POST /cgi-bin/auth HTTP/1.1\r\n'+ \
                  'Host: foo.example\r\n'+ \
                  'Content-Type: application/x-www-form-urlencoded\r\n'+ \
                  f'Content-Length: {len(body)}\r\n'+ \
                  '\r\n'+ \
                  body

        request = self.pad_msg(request.encode(), self.SIZE_POST_REQUEST)
        self.sd.send(request)

        #Wait for response of login
        response_header = ''

        while(True):
            char = self.sd.recv(1).decode('utf-8', 'ignore')
            response_header += char

            if response_header.endswith('\r\n\r\n'):
                break

        response_header = response_header[:-4]
        header_list = response_header.split('\r\n')
        status = header_list[0].split(' ')
        headers = {x.split(': ', 1)[0]:x.split(': ', 1)[1] for x in header_list[1:]}

        #Check how POST request is
        if status[1] == '200':
            #The server was able to access to the DB
            length = int(headers['Content-Length'])
            body = self.sd.recv(length).decode('utf-8', 'ignore')
            n = len(response_header)+4+length
            padding = self.sd.recv(self.SIZE_POST_RESPONSE-n)
            return body
        else:
            #Wrong format of POST request
            n = len(response_header)+4
            padding = self.sd.recv(self.SIZE_POST_RESPONSE-n)
            return 'Some error occurs'

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

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Close the socket stream
        """
        
        self.sd.close()