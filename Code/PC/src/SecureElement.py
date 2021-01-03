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

    def __init__(self, IP_address, port):
        with open('../dat/crypto/ecdsa.key', "r") as sk_file:
            sk_pem = sk_file.read().encode()
            self.ECDSA_PRIVATE_KEY = ecdsa.SigningKey.from_pem(sk_pem)
    
        #print(colored('PRIVATE KEY length: ', 'blue')+str(len(self.ECDSA_PRIVATE_KEY)))
        #cprint(self.ECDSA_PRIVATE_KEY, 'green')

        self.sd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.IP_ADDRESS = IP_address
        self.PORT = port
        
    def __enter__(self):
        self.sd = ssl.wrap_socket(self.sd,
                                  server_side=False,
                                  ca_certs = "../dat/crypto/server.pem", 
                                  certfile="../dat/crypto/client.pem",
                                  keyfile="../dat/crypto/client.key",
                                  cert_reqs=ssl.CERT_REQUIRED,
                                  ssl_version=ssl.PROTOCOL_TLSv1_2)

        self.sd.connect((self.IP_ADDRESS, self.PORT))

        return self

    def sign(self, msg):
        cprint(msg, 'cyan')
        nonce=uuid.uuid4().bytes
        cprint(nonce, 'blue')
        #hash_msg = hashlib.sha256().hexdigest()
        signature = self.ECDSA_PRIVATE_KEY.sign(msg.encode()+nonce)
        cprint(signature, 'green')
        self.sd.send(msg.encode()+b'\r\n'+nonce+signature)
        #Wait for AcCAPPCHA response
        check = ''

        while(True):
            char = self.sd.recv(1).decode('utf-8', 'ignore')
            check += char

            if check.endswith('\r\n'):
                break

        #Return True/False
        if check[:-2] == 'OK':
            return True
        else:
            return False


    def credentials(self, username, password):
        #Send credentials
        hash_pwd = hashlib.sha512(password.encode()).hexdigest()
        body = f'user={username}&pwd={hash_pwd}'

        request = 'POST /cgi-bin/auth HTTP/1.1\r\n'+ \
                  'Host: foo.example\r\n'+ \
                  'Content-Type: application/x-www-form-urlencoded\r\n'+ \
                  f'Content-Length: {len(body)}\r\n'+ \
                  '\r\n'+ \
                  body

        self.sd.send(request.encode())

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

        #Return True/False
        if status[1] == '200':
            length = int(headers['Content-Length'])
            body = self.sd.recv(length).decode('utf-8', 'ignore')
            return body
        else:
            return 'Some error occurs'

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.sd.close()

count=0
while(count<3):
    with SecureElement('127.0.0.1', 8080) as s:
        check = s.sign(str(True))        
        cprint(check, 'yellow')

        if check:
            msg = s.credentials('RaffaDNDM', 'hello3')
#            msg = s.credentials('raffaeledndm', 'ciao')
    
            with open('../dat/html/response.html', 'w') as f:
                f.write(msg)

                import webbrowser, os
                webbrowser.open('file://' + os.path.abspath('../dat/html/response.html'))
                
            #print(msg)
            #msg = s.credentials('RaffaDNDM', 'hello35')
            #msg = s.credentials('JohnSM', 'password4')
            #msg = s.credentials('CristiFB', 'byebye12')
            #msg = s.credentials('IreneRMN', 'flower10')
            break
        else:
            count+=1