import ecdsa
import socket
import uuid
from termcolor import cprint, colored
import hashlib

class SecureElement:
    # SECP256k1 is the Bitcoin elliptic curve
    #PRIVATE_KEY = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1).to_string().hex()
    #PUBLIC_KEY = PRIVATE_KEY.get_verifying_key().to_string().hex()
    PRIVATE_KEY = '86d2c50e3e273a8c31a08cabac7d416d54868d07b756a3b263096ea87da90b1a'
    PUBLIC_KEY = '9aedaaa1468e178426e7cfb7257bee4ee589106ded1f282bc2cde90ebaea07ca'+ \
                 '83ff329740d42624069f77fa360a6c30d2fc26ed7ce9cc32ed97cd5865aaa3c6'
    
    def __init__(self, IP_address, port):
        print(colored('PRIVATE KEY length: ', 'blue')+str(len(self.PRIVATE_KEY)))
        print(colored(' PUBLIC KEY length: ', 'blue')+str(len(self.PUBLIC_KEY)))
        #cprint(self.PRIVATE_KEY, 'green')
        #cprint(self.PUBLIC_KEY, 'red')

        self.sd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.IP_ADDRESS = IP_address
        self.PORT = port
        
    def __enter__(self):
        self.sd.connect((self.IP_ADDRESS, self.PORT))

        return self

    def sign(self, message):
        #Bytes object (sig)
        sign_key = ecdsa.SigningKey.from_string(bytes.fromhex(self.PRIVATE_KEY), curve=ecdsa.SECP256k1)
        cprint(message, 'cyan')
        nonce=uuid.uuid4().bytes
        cprint(nonce, 'blue')
        hash_msg = hashlib.sha256(message.encode()+nonce).hexdigest()
        signature = sign_key.sign(hash_msg.encode())
        cprint(signature, 'green')
        self.sd.send(message.encode()+b'\r\n'+nonce+signature)
        
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.sd.close()


with SecureElement('127.0.0.1', 8080) as s:
    s.sign('ciao')