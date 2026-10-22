import regex as re

corpus_test="""
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
"""
class BPETokenizer:
    def __init__(self,end_token='<|endoftext|>'):
        self.end_token=end_token
        self.PRETOKENIZER_PATH=PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.vocabulary=[self.end_token]+[chr(n) for n in range(256)]
        self.pretok_counter={}

    def pretokenization(self,text):
        matches=re.finditer(self.PRETOKENIZER_PATH,text)
        for m in matches:
            enc_tuple=tuple(bytes([b]) for b in m.group().encode())
            self.pretok_counter[enc_tuple]=self.pretok_counter.get(enc_tuple,0)+1

    def merges(self,n=1,n_max=6):
        bytes_pair_counter={}
        mapper_bp_to_pretok={}
        for key,count in self.pretok_counter.items():
            for byte_pair in list(zip(key[:-1],key[1:])):
                bytes_pair_counter[byte_pair]=bytes_pair_counter.get(byte_pair,0)+count
                list_pretok=mapper_bp_to_pretok.get(byte_pair,list())
                list_pretok.append(key)
                mapper_bp_to_pretok[byte_pair]=list_pretok
        max_bp=max(bytes_pair_counter,key=lambda x: (bytes_pair_counter[x],x))
        print(n,max_bp)
        self.vocabulary.append(b"".join(max_bp).decode())
        for pretok in set(mapper_bp_to_pretok[max_bp]):
            new_pretok=[]
            i=0
            while i<len(pretok)-1:
                if pretok[i]==max_bp[0] and pretok[i+1]==max_bp[1]:
                    new_pretok.append(b"".join(max_bp))
                    i+=2
                else:
                    new_pretok.append(pretok[i])
                    i+=1
            if i==len(pretok)-1: new_pretok.append(pretok[i]) # don't miss the last char

            self.pretok_counter[tuple(new_pretok)]=self.pretok_counter[pretok]
            del self.pretok_counter[pretok]
        print(self.pretok_counter)
        if n<n_max:
            return self.merges(n+1)
        else: 
            return None


            



tokenizer=BPETokenizer()
tokenizer.pretokenization(corpus_test)
counter=tokenizer.pretok_counter
for key,value in counter.items():
    print(key,b"".join(key).decode(),'->',value)

tokenizer.merges()

counter=tokenizer.pretok_counter
for key,value in counter.items():
    print(key,b"".join(key).decode(),'->',value)