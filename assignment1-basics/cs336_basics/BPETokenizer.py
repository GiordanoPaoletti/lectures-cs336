import regex as re
from .pretokenization import find_chunk_boundaries, pretokenization_single
from multiprocessing import Pool,cpu_count

class BPETokenizer:
    def __init__(self,end_token='<|endoftext|>'):
        self.end_token=end_token
        self.PRETOKENIZER_PATH=PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.vocabulary={n:bytes((n,)) for n in range(256)}
        self.vocabulary[len(self.vocabulary)]=self.end_token.encode()
        self.pretok_counter={}

    def pretokenization(self,INPUT_PATH,n_work=None):
        # TODO: estimate peak memory per chunk on OpenWebText (~11GB): do I need more chunks than workers?

        if n_work is None:
            n_work=cpu_count()
        special_token=self.end_token.encode()
        with open(INPUT_PATH,'rb') as f:
            boundaries = find_chunk_boundaries(f, n_work, special_token)
        tasks=[(INPUT_PATH, start, end, self.PRETOKENIZER_PATH, special_token) for start,end in zip(boundaries[:-1], boundaries[1:])]
        with Pool(cpu_count()) as pool:
            for chunk_pretok_counter in pool.imap_unordered(pretokenization_single,tasks):
                if len(self.pretok_counter)==0:
                    self.pretok_counter.update(chunk_pretok_counter)
                else:
                    for key,value in chunk_pretok_counter.items():
                      self.pretok_counter[key]= self.pretok_counter.get(key,0)+value

        
    def merges(self,n=1,n_max=6):
        # TODO: store the `merges` list (tuple[bytes, bytes]) in order of creation
        # TODO: stop based on vocab_size instead of n_max (256 bytes + special tokens count too)
        # TODO: recursion -> with ~10k merges do I hit Python's recursion limit? (and n_max is not propagated)

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
        self.vocabulary[len(self.vocabulary)]=b"".join(max_bp)
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


            
corpus_test="""
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
"""



if __name__ == "__main__":
    tokenizer=BPETokenizer()
    tokenizer.pretokenization('./data/TinyStoriesV2-GPT4-valid.txt')
    counter=tokenizer.pretok_counter
    for key,value in counter.items():
        print(key,b"".join(key).decode(),'->',value)

    # tokenizer.merges()

    # # counter=tokenizer.pretok_counter
    # # for key,value in counter.items():
    # #     print(key,b"".join(key).decode(),'->',value)

    # print(tokenizer.vocabulary)