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
        self.sd.connect((self.IP_ADDRESS, self.PORT))
        self.sd = ssl.wrap_socket(self.sd,
                                  server_side=False,
                                  ca_certs = "../dat/crypto/server.pem", 
                                  certfile="../dat/crypto/client.pem",
                                  keyfile="../dat/crypto/client.key",
                                  cert_reqs=ssl.CERT_REQUIRED,
                                  ssl_version=ssl.PROTOCOL_TLSv1_2)

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
        if check=='OK\r\n':
            return True
        else:
            return False


    def credentials(self, username, password):
        #Send credentials
        hash_msg = hashlib.sha512(password.encode()).hexdigest()
        print(len(hash_msg))
        print(hash_msg)
        #Wait for response of login

        #Return True/False to be used to count trials in AcCAPPCHA
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.sd.close()

count=0
while(count<3):
    with SecureElement('127.0.0.1', 8080) as s:
        check = s.sign(str(True))
        if check:
            s.credentials('RaffaDNDM', 'hello35')
            #s.credentials('JohnSM', 'password4')
            #s.credentials('CristiFB', 'byebye12')
            #s.credentials('IreneRMN', 'flower10')
            break
        else:
            count+=1