import ecdsa
import socket
import uuid


class SecureElement:

    def __init__(self, IP_address, port):
        # SECP256k1 is the Bitcoin elliptic curve
        self.PRIVATE_KEY = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1) 
        self.PUBLIC_KEY = self.PRIVATE_KEY.get_verifying_key()
        print(len(self.PUBLIC_KEY.to_string()))
        print(self.PUBLIC_KEY.to_string())

        self.sd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.IP_ADDRESS = IP_address
        self.PORT = port
        
    def __enter__(self):
        self.sd.connect((self.IP_ADDRESS, self.PORT))

        return self

    def sign(self, response):
        self.sd.send(self.PUBLIC_KEY.to_string())
        msg = self.sd.recv(4).decode('utf-8', 'ignore')
        
        if msg!='OK\r\n':
            print('No correct sent of public key')
            self.__exit__(None, None, None)

        #Bytes object (sig)
        sig = self.PRIVATE_KEY.sign(response.encode())
        self.sd.send(sig)
        
        print(uuid.uuid4())
        
        try:
            return self.PUBLIC_KEY.verify(sig, b'ciao')
        except ecdsa.keys.BadSignatureError:
            return False

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.sd.close()


with SecureElement('127.0.0.1', 8080) as s:
    print(s.sign('ciao'))