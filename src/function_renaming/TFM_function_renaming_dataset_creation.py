import importlib
import time
import pickle
import traceback
import random
import os
import copy
import re
import sys
import json
import yaml 
from pprint import pprint
import numpy as np
from numpy.random import choice
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from random import shuffle
import math


import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import degree



class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

def sanitize(json_str):

    if json_str[0] != '{':
        json_str ='{ '+json_str 

    # if json_str.find('content')>-1:
    #     # extract content
    #     i = json_str.find('content')
    #     # extract what is inside the double quotes of 
    #     # the value of key content

    #     i1 = i + 11
    #     i2 = json_str.find('}') #len(json_str) - 3
    #     thecontent = json_str[i1:i2]
    #     i3 = thecontent.rfind('"')
    #     if i1+i3 != i2-1 and \
    #        i1+i3 != i2-2 :
    #        i3=i2-2

    #     thecontent=thecontent[:i3]

    #     # change back to single quotes
    #     # backshaled single quotes into single quotes
    #     # remove single backslashes
    #     json_str = json_str[:i1] + \
    #                thecontent.replace('"','\'').replace('\\\'','\'').replace('\\','') + \
    #                json_str[i1+i3:]

    # j = json_str.find('}')
    # if j==-1:
    #     json_str = json_str + '}'

    # json_str = json_str[:j+1]

    # if j>-1 and json_str[j-1:j+1]!='"}':
    #     print(json_str)
    #     json_str = json_str+'"}'
    #     print(json_str)

    return """""" + json_str

def readGraph(folder, filename):
    """
    # read nodes, edges and  edge attributes
    use networkx to read edge list txt file 
                 read nodes txt file, 
                 parse for each line the node attrbiutes and add them to the graph 
    """
    g = nx.read_edgelist(folder+filename)
    # read node attributes
    with open(folder+filename.replace('edges','nodes'),'r') as f:
        for line in f.readlines():
            firstsep = line.find("{")
            if firstsep==-1:
                continue

            tokens = line[firstsep:]
            nodeid = line[:firstsep].replace(' ','')
            
            try:
                attr_dict = json.loads(tokens)
            except Exception as err:
                traceback.print_exc()
                print("JSON error with ", tokens)
            
            if nodeid not in g.node.keys():
                g.add_node(nodeid)
            
            for k,v in attr_dict.items():
                try:
                    if len(g.node.keys() )==0:
                        g.node[nodeid]={}
                    g.node[nodeid][k] = v
                except Exception as err:
                    print("problem with ",nodeid,attr_dict,str(err))
                    print("{} in g.node.keys() ?".format(nodeid),nodeid in g.node.keys())
                    #print(filename)
                    traceback.print_exc()
                    
                
    return g

def prepareLabelDict(g, attr):
    """
    recover a specific attribute from each node , save them in a key(nodeid) value dict.
    """

    label_dict = {}
    for node in g.nodes():
        if attr == 'all':
            label_dict[node]=node + " " + \
                             g.node[node]['content']
        else:
            label_dict[node]=g.node[node][attr]
    return label_dict

def plotGraph(g, label=None):
    """
         plot graph with nx.daw() function
         coloring each node by it's type attribute value.
    """
    
    # color plot
    attrs={}
    palette = ['cyan', 'orange','yellow',
               'red','blue', 'grey',
               'magenta','green','black',   
               'pink',]
    types =['instr','register','immediate','memory',
        'displacement','phrase','func','unknown']
    
    for node in g.nodes():
        #if 'type' in g.node[node].keys():
        #    i = types.index(g.node[node]['type'])
        #else:
        if 'type' in g.node[node].keys():
            if g.node[node]['type'] in types:
                i = types.index(g.node[node]['type'])
            else:
                i=len(types)-1
                for j in types:
                    if g.node[node]['type'].startswith(j):
                        i = types.index(j)
        else:
            i=len(palette)-1
        attrs[node]={'color':palette[ i % len(palette)]}

    nx.set_node_attributes(g, attrs)
    colors = nx.get_node_attributes(g, 'color')
    if label:
        
        label_dict= prepareLabelDict(g,attr=label)
        nx.draw(g, 
            node_color=colors.values(),
            labels=label_dict,
            with_labels = True  )
    #nx.draw_circular(g, node_color=colors.values())
    #plt.draw()
    else:
        nx.draw(g, 
            node_color=colors.values())
    
def plotGraphFunction(folder, filename, label=None):
    """
    read fraph and plot it
    """
    g = readGraph(folder, filename)
    plotGraph(g,label)

def read_config_file():

    conf = None
    try:
        with open(os.path.abspath(os.path.join(os.path.abspath('.'),'config.yaml')),'r') as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
        # with open('config.yaml','r') as f:
        #     conf = yaml.load(f, Loader=yaml.FullLoader)
    except:
        pass 

    return conf
        
class FunctionsDataset(Dataset):
    """
        class that contains the methods to 
                read a folder with nx formated graphs
                and create a Pytorch Geometric on disk dataset
    """
    node_translation = {}  # translates from nodeid like im34,r2 to an autoincremented integer
    nodeidmax = -1
    only_graph_features = False

    conf = read_config_file()
    instr_types = conf['instr_types']
    #x86_instr_set = [ i.lower() for i in
    x86_instr_set = conf['x86_instr_set']
    bow_vocab = conf['bow_vocab']

    # instr_types = ['instr','register','immediate','memory',
    #     'displacement','phrase','func','unknown']
    # x86_instr_set = [ i.lower() for i in ['AAA', 'AAD', 'AAM', 'AAS', 'ADC', 'ADCX', 'ADD', 
    #                  'ADDPD', 'ADDPS', 'ADDSD', 'ADDSS', 'ADDSUBPD', 
    #                  'ADDSUBPS', 'ADOX', 'AESDEC', 'AESDECLAST', 'AESENC',
    #                  'AESENCLAST', 'AESIMC', 'AESKEYGENASSIST', 'AND', 'ANDN', 
    #                  'ANDNPD', 'ANDNPS', 'ANDPD', 'ANDPS', 'ARPL', 'BEXTR', 
    #                  'BLENDPD', 'BLENDPS', 'BLENDVPD', 'BLENDVPS', 'BLSI', 
    #                  'BLSMSK', 'BLSR', 'BNDCL', 'BNDCN', 'BNDCU', 'BNDLDX', 
    #                  'BNDMK', 'BNDMOV', 'BNDSTX', 'BOUND', 'BSF', 'BSR', 'BSWAP', 'BT', 'BTC', 'BTR', 'BTS', 'BZHI', 'CALL', 'CBW', 'CDQ', 'CDQE', 'CLAC', 'CLC', 'CLD', 'CLDEMOTE', 'CLFLUSH', 'CLFLUSHOPT', 'CLI', 'CLTS', 'CLWB', 'CMC', 'CMOVcc', 'CMP', 'CMPPD', 'CMPPS', 'CMPS', 'CMPSB', 'CMPSD', 'CMPSQ', 'CMPSS', 'CMPSW', 'CMPXCHG', 'COMISD', 'COMISS', 'CPUID', 'CQO', 'CWD', 'CWDE', 'DAA', 'DAS', 'DEC', 'DIV', 'DIVPD', 'DIVPS', 'DIVSD', 'DIVSS', 'DPPD', 'DPPS', 'EMMS', 'ENTER', 'EXTRACTPS', 'FABS', 'FADD', 'FADDP', 'FBLD', 'FBSTP', 'FCHS', 'FCLEX', 'FCMOVcc', 'FCOM', 'FCOMI', 'FCOMIP', 'FCOMP', 'FCOMPP', 'FCOS', 'FDECSTP', 'FDIV', 'FDIVP', 'FDIVR', 'FDIVRP', 'FFREE', 'FIADD', 'FICOM', 'FICOMP', 'FIDIV', 'FIDIVR', 'FILD', 'FIMUL', 'FINCSTP', 'FINIT', 'FIST', 'FISTP', 'FISTTP', 'FISUB', 'FISUBR', 'FLD', 'FLDCW', 'FLDENV', 'FLDPI', 'FLDZ', 'FMUL', 'FMULP', 'FNCLEX', 'FNINIT', 'FNOP', 'FNSAVE', 'FNSTCW', 'FNSTENV', 'FNSTSW', 'FPATAN', 'FPREM', 'FPTAN', 'FRNDINT', 'FRSTOR', 'FSAVE', 'FSCALE', 'FSIN', 'FSINCOS', 'FSQRT', 'FST', 'FSTCW', 'FSTENV', 'FSTP', 'FSTSW', 'FSUB', 'FSUBP', 'FSUBR', 'FSUBRP', 'FTST', 'FUCOM', 'FUCOMI', 'FUCOMIP', 'FUCOMP', 'FUCOMPP', 'FWAIT', 'FXAM', 'FXCH', 'FXRSTOR', 'FXSAVE', 'FXTRACT', 'HADDPD', 'HADDPS', 'HLT', 'HSUBPD', 'HSUBPS', 'IDIV', 'IMUL', 'IN', 'INC', 'INS', 'INSB', 'INSD', 'INSERTPS', 'INSW', 'INTO', 'INVD', 'INVLPG', 'INVPCID', 'IRET', 'IRETD', 'JMP', 'Jcc', 'KADDB', 'KADDD', 'KADDQ', 'KADDW', 'KANDB', 'KANDD', 'KANDNB', 'KANDND', 'KANDNQ', 'KANDNW', 'KANDQ', 'KANDW', 'KMOVB', 'KMOVD', 'KMOVQ', 'KMOVW', 'KNOTB', 'KNOTD', 'KNOTQ', 'KNOTW', 'KORB', 'KORD', 'KORQ', 'KORTESTB', 'KORTESTD', 'KORTESTQ', 'KORTESTW', 'KORW', 'KSHIFTLB', 'KSHIFTLD', 'KSHIFTLQ', 'KSHIFTLW', 'KSHIFTRB', 'KSHIFTRD', 'KSHIFTRQ', 'KSHIFTRW', 'KTESTB', 'KTESTD', 'KTESTQ', 'KTESTW', 'KUNPCKBW', 'KUNPCKDQ', 'KUNPCKWD', 'KXNORB', 'KXNORD', 'KXNORQ', 'KXNORW', 'KXORB', 'KXORD', 'KXORQ', 'KXORW', 'LAHF', 'LAR', 'LDDQU', 'LDMXCSR', 'LDS', 'LEA', 'LEAVE', 'LES', 'LFENCE', 'LFS', 'LGDT', 'LGS', 'LIDT', 'LLDT', 'LMSW', 'LOCK', 'LODS', 'LODSB', 'LODSD', 'LODSQ', 'LODSW', 'LOOP', 'LOOPcc', 'LSL', 'LSS', 'LTR', 'LZCNT', 'MASKMOVDQU', 'MASKMOVQ', 'MAXPD', 'MAXPS', 'MAXSD', 'MAXSS', 'MFENCE', 'MINPD', 'MINPS', 'MINSD', 'MINSS', 'MONITOR', 'MOV', 'MOVAPD', 'MOVAPS', 'MOVBE', 'MOVD', 'MOVDDUP', 'MOVDIRI', 'MOVDQA', 'MOVDQU', 'MOVHLPS', 'MOVHPD', 'MOVHPS', 'MOVLHPS', 'MOVLPD', 'MOVLPS', 'MOVMSKPD', 'MOVMSKPS', 'MOVNTDQ', 'MOVNTDQA', 'MOVNTI', 'MOVNTPD', 'MOVNTPS', 'MOVNTQ', 'MOVQ', 'MOVS', 'MOVSB', 'MOVSD', 'MOVSHDUP', 'MOVSLDUP', 'MOVSQ', 'MOVSS', 'MOVSW', 'MOVSX', 'MOVSXD', 'MOVUPD', 'MOVUPS', 'MOVZX', 'MPSADBW', 'MUL', 'MULPD', 'MULPS', 'MULSD', 'MULSS', 'MULX', 'MWAIT', 'NEG', 'NOP', 'NOT', 'OR', 'ORPD', 'ORPS', 'OUT', 'OUTS', 'OUTSB', 'OUTSD', 'OUTSW', 'PABSB', 'PABSD', 'PABSQ', 'PABSW', 'PACKSSDW', 'PACKSSWB', 'PACKUSDW', 'PACKUSWB', 'PADDB', 'PADDD', 'PADDQ', 'PADDSB', 'PADDSW', 'PADDUSB', 'PADDUSW', 'PADDW', 'PALIGNR', 'PAND', 'PANDN', 'PAUSE', 'PAVGB', 'PAVGW', 'PBLENDVB', 'PBLENDW', 'PCLMULQDQ', 'PCMPEQB', 'PCMPEQD', 'PCMPEQQ', 'PCMPEQW', 'PCMPESTRI', 'PCMPESTRM', 'PCMPGTB', 'PCMPGTD', 'PCMPGTQ', 'PCMPGTW', 'PCMPISTRI', 'PCMPISTRM', 'PDEP', 'PEXT', 'PEXTRB', 'PEXTRD', 'PEXTRQ', 'PEXTRW', 'PHADDD', 'PHADDSW', 'PHADDW', 'PHMINPOSUW', 'PHSUBD', 'PHSUBSW', 'PHSUBW', 'PINSRB', 'PINSRD', 'PINSRQ', 'PINSRW', 'PMADDUBSW', 'PMADDWD', 'PMAXSB', 'PMAXSD', 'PMAXSQ', 'PMAXSW', 'PMAXUB', 'PMAXUD', 'PMAXUQ', 'PMAXUW', 'PMINSB', 'PMINSD', 'PMINSQ', 'PMINSW', 'PMINUB', 'PMINUD', 'PMINUQ', 'PMINUW', 'PMOVMSKB', 'PMOVSX', 'PMOVZX', 'PMULDQ', 'PMULHRSW', 'PMULHUW', 'PMULHW', 'PMULLD', 'PMULLQ', 'PMULLW', 'PMULUDQ', 'POP', 'POPA', 'POPAD', 'POPCNT', 'POPF', 'POPFD', 'POPFQ', 'POR', 'PREFETCHW', 'PREFETCHh', 'PSADBW', 'PSHUFB', 'PSHUFD', 'PSHUFHW', 'PSHUFLW', 'PSHUFW', 'PSIGNB', 'PSIGND', 'PSIGNW', 'PSLLD', 'PSLLDQ', 'PSLLQ', 'PSLLW', 'PSRAD', 'PSRAQ', 'PSRAW', 'PSRLD', 'PSRLDQ', 'PSRLQ', 'PSRLW', 'PSUBB', 'PSUBD', 'PSUBQ', 'PSUBSB', 'PSUBSW', 'PSUBUSB', 'PSUBUSW', 'PSUBW', 'PTEST', 'PTWRITE', 'PUNPCKHBW', 'PUNPCKHDQ', 'PUNPCKHQDQ', 'PUNPCKHWD', 'PUNPCKLBW', 'PUNPCKLDQ', 'PUNPCKLQDQ', 'PUNPCKLWD', 'PUSH', 'PUSHA', 'PUSHAD', 'PUSHF', 'PUSHFD', 'PUSHFQ', 'PXOR', 'RCL', 'RCPPS', 'RCPSS', 'RCR', 'RDFSBASE', 'RDGSBASE', 'RDMSR', 'RDPID', 'RDPKRU', 'RDPMC', 'RDRAND', 'RDSEED', 'RDTSC', 'RDTSCP', 'REP', 'REPE', 'REPNE', 'REPNZ', 'REPZ', 'RET', 'ROL', 'ROR', 'RORX', 'ROUNDPD', 'ROUNDPS', 'ROUNDSD', 'ROUNDSS', 'RSM', 'RSQRTPS', 'RSQRTSS', 'SAHF', 'SAL', 'SAR', 'SARX', 'SBB', 'SCAS', 'SCASB', 'SCASD', 'SCASW', 'SETcc', 'SFENCE', 'SGDT', 'SHL', 'SHLD', 'SHLX', 'SHR', 'SHRD', 'SHRX', 'SHUFPD', 'SHUFPS', 'SIDT', 'SLDT', 'SMSW', 'SQRTPD', 'SQRTPS', 'SQRTSD', 'SQRTSS', 'STAC', 'STC', 'STD', 'STI', 'STMXCSR', 'STOS', 'STOSB', 'STOSD', 'STOSQ', 'STOSW', 'STR', 'SUB', 'SUBPD', 'SUBPS', 'SUBSD', 'SUBSS', 'SWAPGS', 'SYSCALL', 'SYSENTER', 'SYSEXIT', 'SYSRET', 'TEST', 'TPAUSE', 'TZCNT', 'UCOMISD', 'UCOMISS', 'UD', 'UMONITOR', 'UMWAIT', 'UNPCKHPD', 'UNPCKHPS', 'UNPCKLPD', 'UNPCKLPS', 'VALIGND', 'VALIGNQ', 'VBLENDMPD', 'VBLENDMPS', 'VBROADCAST', 'VCOMPRESSPD', 'VCOMPRESSPS', 'VDBPSADBW', 'VERR', 'VERW', 'VEXPANDPD', 'VEXPANDPS', 'VFIXUPIMMPD', 'VFIXUPIMMPS', 'VFIXUPIMMSD', 'VFIXUPIMMSS', 'VFPCLASSPD', 'VFPCLASSPS', 'VFPCLASSSD', 'VFPCLASSSS', 'VGATHERDPD', 'VGATHERDPS', 'VGATHERQPD', 'VGATHERQPS', 'VGETEXPPD', 'VGETEXPPS', 'VGETEXPSD', 'VGETEXPSS', 'VGETMANTPD', 'VGETMANTPS', 'VGETMANTSD', 'VGETMANTSS', 'VMASKMOV', 'VPBLENDD', 'VPBLENDMB', 'VPBLENDMD', 'VPBLENDMQ', 'VPBLENDMW', 'VPBROADCAST', 'VPBROADCASTB', 'VPBROADCASTD', 'VPBROADCASTM', 'VPBROADCASTQ', 'VPBROADCASTW', 'VPCMPB', 'VPCMPD', 'VPCMPQ', 'VPCMPUB', 'VPCMPUD', 'VPCMPUQ', 'VPCMPUW', 'VPCMPW', 'VPCOMPRESSD', 'VPCOMPRESSQ', 'VPCONFLICTD', 'VPCONFLICTQ', 'VPERMB', 'VPERMD', 'VPERMILPD', 'VPERMILPS', 'VPERMPD', 'VPERMPS', 'VPERMQ', 'VPERMW', 'VPEXPANDD', 'VPEXPANDQ', 'VPGATHERDD', 'VPGATHERDQ', 'VPGATHERQD', 'VPGATHERQQ', 'VPLZCNTD', 'VPLZCNTQ', 'VPMASKMOV', 'VPMOVDB', 'VPMOVDW', 'VPMOVQB', 'VPMOVQD', 'VPMOVQW', 'VPMOVSDB', 'VPMOVSDW', 'VPMOVSQB', 'VPMOVSQD', 'VPMOVSQW', 'VPMOVSWB', 'VPMOVUSDB', 'VPMOVUSDW', 'VPMOVUSQB', 'VPMOVUSQD', 'VPMOVUSQW', 'VPMOVUSWB', 'VPMOVWB', 'VPMULTISHIFTQB', 'VPROLD', 'VPROLQ', 'VPROLVD', 'VPROLVQ', 'VPRORD', 'VPRORQ', 'VPRORVD', 'VPRORVQ', 'VPSCATTERDD', 'VPSCATTERDQ', 'VPSCATTERQD', 'VPSCATTERQQ', 'VPSLLVD', 'VPSLLVQ', 'VPSLLVW', 'VPSRAVD', 'VPSRAVQ', 'VPSRAVW', 'VPSRLVD', 'VPSRLVQ', 'VPSRLVW', 'VPTERNLOGD', 'VPTERNLOGQ', 'VPTESTMB', 'VPTESTMD', 'VPTESTMQ', 'VPTESTMW', 'VPTESTNMB', 'VPTESTNMD', 'VPTESTNMQ', 'VPTESTNMW', 'VRANGEPD', 'VRANGEPS', 'VRANGESD', 'VRANGESS', 'VREDUCEPD', 'VREDUCEPS', 'VREDUCESD', 'VREDUCESS', 'VRNDSCALEPD', 'VRNDSCALEPS', 'VRNDSCALESD', 'VRNDSCALESS', 'VSCALEFPD', 'VSCALEFPS', 'VSCALEFSD', 'VSCALEFSS', 'VSCATTERDPD', 'VSCATTERDPS', 'VSCATTERQPD', 'VSCATTERQPS', 'VTESTPD', 'VTESTPS', 'VZEROALL', 'VZEROUPPER', 'WAIT', 'WBINVD', 'WRFSBASE', 'WRGSBASE', 'WRMSR', 'WRPKRU', 'XABORT', 'XACQUIRE', 'XADD', 'XBEGIN', 'XCHG', 'XEND', 'XGETBV', 'XLAT', 'XLATB', 'XOR', 'XORPD', 'XORPS', 'XRELEASE', 'XRSTOR', 'XRSTORS', 'XSAVE', 'XSAVEC', 'XSAVEOPT', 'XSAVES', 'XSETBV', 'XTEST', 'EACCEPT', 'EACCEPTCOPY', 'EADD', 'EAUG', 'EBLOCK', 'ECREATE', 'EDBGRD', 'EDBGWR', 'EDECVIRTCHILD', 'EENTER', 'EEXIT', 'EEXTEND', 'EGETKEY', 'EINCVIRTCHILD', 'EINIT', 'ELBUC', 'ELDB', 'ELDBC', 'ELDU', 'EMODPE', 'EMODPR', 'EMODT', 'ENCLS', 'ENCLU', 'ENCLV', 'EPA', 'ERDINFO', 'EREMOVE', 'EREPORT', 'ERESUME', 'ESETCONTEXT', 'ETRACK', 'ETRACKC', 'EWB', 'INVEPT', 'INVVPID', 'VMCALL', 'VMCLEAR', 'VMFUNC', 'VMLAUNCH', 'VMPTRLD', 'VMPTRST', 'VMREAD', 
    #                  'VMRESUME', 'VMWRITE', 'VMXOFF', 'VMXON']]
    
    myprocessed_filenames = []
    problematic_files = []
    my_classes = set([])
    
    def __init__(self, root, transform=None, pre_transform=None):
        super(FunctionsDataset, self).__init__(root, transform, pre_transform)
        #print(" end of __init__()")
        #print(self.bow_vocab[:10])
        #print(self.instr_types)

    @property
    def raw_file_names(self):
        return ['graphs01']

    @property
    def processed_file_names(self):
        """
        returns the list of processed filenames.
                The first time it is called, it will update the self.myprocessed_filenames.

        #return self.myprocessed_filenames
        # update the list of processed filenames
        # assumes base dir?? how to resolve that?
        

        """
        if len(self.myprocessed_filenames)==0:

            # if myprocessed_filenames is not initialized,
            # read all filenames from processed folder
            #print(self.bow_vocab[:10])
            #print(self.x86_instr_set[:10])
            #print(self.instr_types)

            processed_path = os.path.join(self.root, 'processed')
            print("processed_path",processed_path)
            self.myprocessed_filenames = []
            for item in os.listdir(processed_path):
                if os.path.isfile(os.path.join(processed_path, item)):
                    self.myprocessed_filenames.append(item)

            #self.processed_paths = self.myprocessed_filenames # can't set attribute
            return self.myprocessed_filenames
        else:
            return self.myprocessed_filenames
    
    @property
    def num_classes(self):
        """

        read the num classes by the subdirs in raw/graphs01

        """
        # new approach
        try:
            return len(
                list(
                    set(
                        [self.get(j).y[0].item() for j in range(self.__len__() )]
                        )))
        except:

            # return len(
            #     list(
            #         set(
            #             [self.get(j).y[0] for j in range(self.__len__() )]
            #             )))

            list_classes = []
            for j in range(self.__len__()):
                list_classes.append(self.get(j).y)
            return list(set(list_classes))


        # # old approach
        # k = len(self.my_classes)
        # if k <= 0 or True:
        #     k=0
        #     # read the num classes by the subdirs in raw/graphs01
        #     rawfolder = os.path.join(self.root, 'raw/graphs01')
        #     for item in os.listdir(rawfolder):
        #         if os.path.isdir(os.path.join(rawfolder, item)):
        #             k+=1
        # return k 
    
    @property
    def num_features(self):
        """
        # this should be dynamic!!
        # load one of the processed files and count the features in there!
        """

        graph0 = self.get(0)
        print("num_features, data.x.shape")
        pprint(graph0.x.shape)
        return graph0.x.shape[1]
        #return 4
    
    @property
    def y(self):
        """
        return a torch vector with all y of all graphs inside
        """

        y = torch.zeros(self.__len__(), dtype=torch.float)

        for j in range(self.__len__()):
            y[j]=self.get(j).y

        return y
    

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        """
        # Download to `self.raw_dir`.
        # this step is not implemented. You need to manually
        # copy the folders that contain the output of the plugin
        # inside the raw folder
        """
        print("Not implemented. Missing folders with graph files in txt for Nx format, for each program to be included in the dataset")


    def save_changes(self, idx, data):
        """
            When modifying an attribute,
            for example self.y, 
            this method must be called

            Cleaner way would be to inherit from torch.utils.Data and save the filepath and implement a method save.
        """

        
        
        pointers = []
        if isinstance(idx, int):
            # transform this into filename from a list of processed paths
            
            try:
                #realfile = self.processed_paths[idx]
                realfile = self.myprocessed_filenames[idx]
                filename = os.path.join(self.processed_dir,realfile)
                torch.save(data, filename)
            except:
                #print(idx, "self.processed_file_names:", self.processed_file_names[:10])
                raise IndexError("error getting: ", idx)
        else:
            print("Error not implemented without integer as index")



    def process(self):
        """
          this function must read the datasets
          every graph must create a file and increment an index
          and also save the filename to self.myprocessed_files
        """
        
        # Read data into huge `Data` list.
        start = time.time()
        print(" FunctionsDataset: process()")
        print("self.processed_paths",self.processed_paths[:5])
        print("self.root",self.root)
        print("self.raw_dir",self.raw_dir)
        print("self.raw_file_names",self.raw_file_names[:5])
        print("self.raw_paths",self.raw_paths[:5])
        print()
        
        i=0
        for folder in self.raw_paths:
            # navigate the folder hierarchy
            for root, dirs, files in os.walk(folder, topdown=False):
                for name in files:
                    if i== -1:
                        break
                    # and create ggraph g for each pair of files
                    filename = os.path.join(root, name)
                    if filename.find('_edges') > -1: # take the *_edges file

                        # generate the PyG graph
                        data = self.parseGraph(filename)
                        if data is None:
                            self.problematic_files.append(filename)
                            continue
                            
                        if self.pre_filter is not None and not self.pre_filter(data):
                            self.problematic_files.append(filename)
                            continue
                            
                        if self.pre_transform is not None:
                            data = self.pre_transform(data)

                        newfile = 'data_{}.pt'.format(i)
                        torch.save(data, os.path.join(self.processed_dir, newfile))
                        self.myprocessed_filenames.append(newfile)
                        #print(i,filename)
                        i += 1
        end = time.time()
        print()
        print("Finished reading-processing dataset in", round(end-start),"s")
        print("processed: {} total files, {} created graph files, {} problematic files".format(
            i+len(self.problematic_files),i,len(self.problematic_files)))
        print()


    def shuffle(self):
        """
        # randomize the list of processed files form disk
        CUrrently deactivated
        """
        #shuffle(self.processed_file_names)
        return self
         
    def get(self, idx):
        """

        method to load or slice graphs from the dataset

        # idx is a torch.LongTensor
        # extract a list from the longTensor
        # load each file separately --
        # load then in the final data tensor
        """
        pointers = []
        
        if isinstance(idx, int):
            # transform this into filename from a list of processed paths
            
            try:
                #realfile = self.processed_paths[idx]
                realfile = self.myprocessed_filenames[idx]
                filename = os.path.join(self.processed_dir,realfile)
                data = torch.load(filename)
                if self.only_graph_features:
                    # remove topo/code/document features from data in memory
                    delattr(data,'code_feats')
                    delattr(data,'x_topo_feats')
                    delattr(data,'x_topo_times')
                    delattr(data,'label')
                    delattr(data,'filename')
                    delattr(data,'tfidf_vec')  # instead of useing tfidf_vec we use edge_attr as a workaround
                    if not isinstance(data.y,torch.Tensor):
                        data.y = torch.LongTensor([[data.y]])
                    #print(data.y[0].item())
                    #print(type(data.y))
                    #print(data.y.shape)
                return data
            except:
                #print(idx, "self.processed_file_names:", self.processed_file_names[:10])
                raise IndexError("error getting: ", idx)
        else:
            pointers = idx.tolist()

            graph_tensors = []
            filenames = []
            for pointer in pointers:
                #filename = os.path.join(self.processed_dir,'data_{}.pt'.format(pointer))
                #filename = 'data_{}.pt'.format(pointer)
                filename = self.myprocessed_filenames[pointer]
                try:
                    #print("reading: ",filename)
                    #graph_tensor = torch.load(filename)
                    #graph_tensors.append(graph_tensor)
                    filenames.append(filename)
                except:
                    print("omitting: ",filename)
                    # we should append some empty tensor or anything
                    #pass
                    
            

            # transform into a dataset with num_classes and num_features ...
            return self.create_pyg_dataset(filenames)

    def create_pyg_dataset(self, processed_paths_sub_list):
        """
            creates a slice of a Dataset
                -> simply changes the processed_paths list (myprocessed_filenames)
                to constitute a slice of the original in disk list of processed files.

            data_list contains Data objects
            Goal:
                -return a Dataset object
                -that allows to reference the Data objects
                -has num_classes
                -has num_features
                -anything else?
        """
        #print(" old dataset",self.myprocessed_filenames[:3])
        new_dataset = copy.deepcopy(self)
        new_dataset.myprocessed_filenames = list(processed_paths_sub_list)
        #print(new_dataset.myprocessed_filenames)

        #print(" new dataset",new_dataset.myprocessed_filenames[:3])
        return new_dataset




    def verify_created_graph(self, data):
        """
        some minor verifications for use afeter slicing a dataset
        """

        # rule0: verify only of data is a nx object
        # otherwise return
        if data is None:
            return False

        # rule1: 2 nodes must exist
        if len(data.nodes())<2:
            return False

        # rule2: some edge must exist
        if len(data.edges())<1:
            return False

        return True


    def parseGraph(self, filename):
        """
        COMMENT:
            reads a graph filename, get's it's label and creates a NX graph with target and code features
        CODE: 
            y = extracts class label from folder name,
            g=readGraph()
            g= verify_created_graph()
            data = createGraphFromNxwithTarget
            data = add_code_features()
            return data
        """
        try:
            y = self.extractClassLabelFromFolderName(os.path.dirname(filename))
            g = readGraph('',filename)
            if not self.verify_created_graph(g):
                #print("problem with: ", filename, len(g.nodes()), len(g.edges()))
                return None
            xlen = len(g.nodes())
            #print(filename)
            data = self.createGraphFromNXwithTarget(g,y,xlen,undirected=True, filename=filename)
            data = self.add_code_features(data,'',filename)
            data.filename = filename

            return data
        except Exception as err:
            print(filename)
            traceback.print_exc()
            return None
        
    
    def extractClassLabelFromFolderName(self, root):
        """
        # extract class from folder name
        only used in version 0 dataset (big_noisy_dataset
        """
        class_label = os.path.dirname(root)
        class_label = os.path.basename(class_label)
        y = [0]
        if class_label.lower().startswith("network"):
            y = [0]
            #print("class", y)
        elif class_label.lower().startswith("crypto"):
            y = [1]
        elif class_label.lower().startswith("disk"):
            y = [2]
            #print("class", y)
        elif class_label.lower().startswith("packer"):
            y = [3]
            #print("class", y)
        else:
            #print("Unknown class! ",root,class_label)
            pass

        
        self.my_classes.add(y[0])
        return y
        
    def createGraph(self, x, edge_index):
        """
            Creates a PyTorch Geometric Dataset
            from a torch tensor (node features) x
            and an edge_index (adjacency list in the form
            [[n1, n1, n1, n2, n2, n3], [n1, n3, n2, n3, n1, n1]]
            for the edges (n1,n1), (n1,n3), (n1,n2), (n2, n3), (n2, n1)
            and (n3, n1)
        """
        return Data(x=x, edge_index=edge_index)


    def resetNodeTranslation(self):
        """
            For each function graph, independently, 
            the id's of the nodes is set from 0 to m

            Each time a function and so it's graph is parsed, self.nodeidmax and self.node_translation is reset.
        """
        self.nodeidmax = -1
        self.node_translation = {}
        
    def translateNodeIds(self,nodeid,g):
        """
            nodeid's contain id and str of the type of the node.
            This function translates all node ids of a graph into integers,
            and removes type information from nodeid

            Pre:
                nodeid is a string (usually string but could be anything)
            Post:
                if nodeid has already been translated, it will return the previous translation.
                otherwise it will create a new autoincremented integer and return it and save it
                
                new version:
                    now each type has it's own dictionary with autoincrement
                    at the end, they are appended, and id's are added with an offset
        """
        
        
        # get the type of the node
        # there is a different dictionary with its autoincrement for each nodetype, 
        # (but at the end they are added with an offset)
        node_type = ''
        if 'type' in g.node[nodeid].keys():
            node_type = g.node[nodeid]['type']
            
        if node_type not in self.node_translation.keys():
            self.node_translation[node_type]={'_maxid': -1}
        
        if nodeid in self.node_translation[node_type].keys():
            return self.node_translation[node_type][nodeid], node_type
        else:
            self.node_translation[node_type]['_maxid']+=1
            
            nid = self.node_translation[node_type]['_maxid']
            self.node_translation[node_type][nodeid] = nid
            return nid, node_type
        
    def one_hot_nodetype(self, gnode, typelist = None):
        """ 
            a vector with a 1 in the actual type 
            and 0's in all other types
            
            
        """
        if typelist is None:
            typelist = self.instr_types
        result = [0 for i in range(len(typelist)) ]
        typestring = ''
        if 'type' in gnode.keys():
            typestring = gnode['type'].split('; ')[0]
            try:
                index_type = typelist.index(typestring)
                result[index_type]=1
            except:
                print("one_hot_nodetype, type not found: ", typestring)
        
        return result

    def one_hot_nodetype_v2(self, gnode, typelist = None):
        """ 
            a vector with a 1 in the actual type 
            and 0's in all other types
            
            version2: return index_type instead of vector of 0's and 1's
        """
        if typelist is None:
            typelist = self.instr_types
        typestring = ''
        if 'type' in gnode.keys():
            typestring = gnode['type'].split('; ')[0]
            try:
                index_type = typelist.index(typestring)
                
                return index_type
            except:
                print("one_hot_nodetype, type not found: ", typestring)
        
        return -1
    
    def one_hot_instruction(self, gnode):
        """
            Extracted the listing from wikipedia (may be incomplete)
            
            if node is not an instruction, try will pass and return all 0s


            a vector with a 1 in the actual instr 
            and 0's in all other instr

            version2: return index_mnemonic?
        
        """
        result = [0 for i in range(len(self.x86_instr_set)) ]
        try:
            index_mnemonic = self.x86_instr_set.index(gnode['content'])
            #print(" instruction ", gnode['content']," as ",index_mnemonic)
            result[index_mnemonic] = 1
        except:
            pass
        return result

    def one_hot_instruction_v2(self, gnode):
        """
            Extracted the listing from wikipedia (may be incomplete)
            
            if node is not an instruction, try will pass and return all 0s


            a vector with a 1 in the actual instr 
            and 0's in all other instr

            version2: return index_mnemonic?
        
        """
        #result = [0 for i in range(len(self.x86_instr_set)) ]
        try:
            index_mnemonic = self.x86_instr_set.index(gnode['content'])
            #print(" instruction ", gnode['content']," as ",index_mnemonic)
            result[index_mnemonic] = 1
            return index_mnemonic
        except:
            pass
        return -1
    
    def update_edge_list(self, edge_list_1, n1, node_type):
        """
            edge_list_1 is a dict that contains a list for each type of node
            append node n1 or create a new list with node_type and append n1
        """
        if node_type not in edge_list_1.keys():
            edge_list_1[node_type]=[n1]
            
        else:
            edge_list_1[node_type].append(n1)
            
        return edge_list_1
    
    def merge_edge_list(self, edge_list):
        """
            pre: edge_list a dict of lists of ints

            merge edge_list , by incrementing the element id  values when necessary to avoid colision of ids
        """
        result_list = []
        previous_max_id = 0
        for k,vlist in edge_list.items():
            result_list.extend([ elemid + previous_max_id for elemid in vlist])
            previous_max_id = max(vlist) + previous_max_id
            
        return result_list

    def tensorize_node_attributes(self, x):
        """
            GOAL:
            To use node attributes in PyTorch
            we need them to be translated to float values.
    
            STEPS:
            1. sort nodes and their attributes by the nodeid
            2. make sure all attributes are integers that are less than an arbitrary maximum
            3. convert them to floats
            4. create a torch.tensor

        """
        # now transform into a sorted list
        x = sorted([(k,v) for k,v in x.items()], key = lambda x: x[0] )
        #x = [ e[1] for e in x]
        res = []
        for e in x:
            elem = e[1] # this is the value so [memaddr, content_not_instr, content_instr, instr_typ]
            # protection against very high integers
            # should only happend in content_not_instr
            for i in range(len(e[1])):
                #if e[1][i] >= 18446744073709551612/20: #sys.double_info.max:
                if e[1][i] >= 184467440737095516: # now using a random high number
                    e[1][i] = int(np.log(e[1][i]))
                e[1][i]=float(e[1][i])
            res.append(elem)
        x = res
        #print(x[:2])
        #print(max(x))
        #pprint(x)
        #pprint(edge_index)
        x = torch.tensor(x, dtype=torch.float)
        return x

    def code_attr_memaddres(self, x,nodeid,g, node):
        """
        # memaddres - int
        # node is an int or a string  im32 -> how to separate
        # if it contains any alphabetical number -> no memaddres
        """
        memaddr = -1.0
        if not re.search('[^0-9]', node):
            memaddr = int(node)
                
        x[nodeid].append(memaddr)

    def code_attr_instr_type(self,x,nodeid,g,node):
        """   

            # instr_type - int
            # vector with 0's and a 1
            # version2: replaced with int
        """
        instr_type = self.one_hot_nodetype_v2(g.node[node])
        x[nodeid].append(instr_type)

    def code_attr_content_not_instr(self,x,nodeid,g,node):
        """
         # content_not_instr - int

         this should be rethinked into a global map from 
         string content to int id
           
        """             
        if 'type' in g.node[node].keys() and \
           g.node[node]['type'] != 'instr':
            try:
                content_not_instr = int(g.node[node]['content'])
            except:
                content_not_instr = 0
        else:
            content_not_instr = 0
        x[nodeid].append(content_not_instr)

    def code_attr_content_instr(self,x,nodeid,g,node):
        """           
            # content_instr - int
            # vector with 0's and a 1 in the actual instruction
            # version2: replaced with int

            one_hot_instruction_v2 could use GLOBAL ids, 
            for later better classification maybe
        """
        content_instr = self.one_hot_instruction_v2(g.node[node])
        x[nodeid].append(content_instr)


    def createGraphFromNX(self, g, xlen, undirected=True):
        """
            Creates a PyTorch Geometric dataset
            from a NetworkX graph

            edge list:
                currently there a list of edge for each node_type
                so edge_list_1 and edge_list_2 are dicts of lists of nodes.
                first we translate nodeids in those dicts of lists
                then we merge them to become a 2 list of nodes , forming the edge list (list of origins and list of endpoints)

            node features(attributes):
                starting point: 
                   g.node[node]={
                     'type': 'register' ,
                      'content' : integer}
                  and node as a memaddr
                
                result:
                 x [ memaddrr, type_as_one-hot-enc, float(content), float(node)]

        """


        # edge list preparation
        edges = g.edges
        edge_list_1 = {}
        edge_list_2 = {}
        self.resetNodeTranslation()
        for e in edges:
            # node id must be an int
            
            # make nodeid by type
            # then merge the list
            n1, node_type1 = self.translateNodeIds(e[0],g)
            n2, node_type2 = self.translateNodeIds(e[1],g)
            
            # add node and node type to edge_list1 and 2
            # edge_list is implemented as 2 lists,
            # one holding the origin node for each edge, and one
            # holding the end node for each edge.
            self.update_edge_list(edge_list_1, n1, node_type1)
            self.update_edge_list(edge_list_2, n2, node_type2)
            
            # edge features?
            
            if undirected:
                self.update_edge_list(edge_list_2, n1, node_type1)
                self.update_edge_list(edge_list_1, n2, node_type2)
                # edge features?

        edge_list_1 = self.merge_edge_list(edge_list_1)
        edge_list_2 = self.merge_edge_list(edge_list_2)
        edge_index = torch.tensor([ edge_list_1,
                                    edge_list_2], dtype=torch.long)

        # node and its features(attributes)
        n = xlen
        x = {}
        for node in g.nodes():
            # translate nodeid first
            nodeid = self.translateNodeIds(node, g)
            x[nodeid] = []

            # code graph node attributes
            self.code_attr_memaddres(x,nodeid,g,node)
            self.code_attr_instr_type(x,nodeid,g,node)
            self.code_attr_content_not_instr(x,nodeid,g,node)
            self.code_attr_content_instr(x,nodeid,g,node)

        # transform all attributes into floats and tensorize 
        x = self.tensorize_node_attributes(x)
        return self.createGraph(x, edge_index)


    def avg_degree(self, g):
        """
            avg all degrees of graph nodes
        """
        degrees = dict(g.degree())
        sum_of_edges = sum(degrees.values())
        return sum_of_edges/len(g.nodes)

    def topological_features(self, g, filename=None):
        """
            Set of topological graph features (from NetworkX package)

            They are appended into a list 

            list of times of computation (for statistics and later decision)
        """

        features = []
        times = []

        # add graph features here

        start = time.time()
        try:
            size = nx.number_of_nodes(g)
        except:
            size = 0.0
        features.append(size)
        times.append(time.time() - start)

        start = time.time()
        try:
            diameter = nx.diameter(g)
        except:
            diameter = 0.0
        features.append(diameter)
        times.append(time.time() - start)

        start = time.time()
        try:
            radius = nx.radius(g)
        except:
            radius = 0.0
        features.append(radius)
        times.append(time.time() - start)

        start = time.time()
        try:
            avg_degree = self.avg_degree(g)
        except:
            avg_degree = 0.0
        features.append(avg_degree)
        times.append(time.time() - start)
        #avg_degree_connectivity = nx.average_degree_connectivity(g)
        
        start = time.time()
        try:
            density = nx.classes.function.density(g)
        except:
            density = 0.0
        features.append(density)
        times.append(time.time() - start)

        start = time.time()
        try:
            connectivity = nx.node_connectivity(g)
        except:
            connectivity = 0.0
        features.append(connectivity)
        times.append(time.time() - start)
        
        start = time.time()
        try:
            avg_clustering = nx.average_clustering(g)
        except:
            avg_clustering = 0.0
        features.append(avg_clustering)
        times.append(time.time() - start)

        start = time.time()
        try:
            avg_shortest_path_length = nx.algorithms.shortest_paths.average_shortest_path_length(g)
        except:
            avg_shortest_path_length = 0.0
        features.append(avg_shortest_path_length)
        times.append(time.time() - start)

        #degree_centrality = nx.algorithms.centrality.degree_centrality(g)
        start = time.time()
        try:
            degree_assortativity = nx.algorithms.assortativity.degree_assortativity_coefficient(g)
            if math.isnan(degree_assortativity):
                degree_assortativity = 0.0
        except Exception as err:
            traceback.print_exc()
            print("Assortativity error with  file:", filename)
            degree_assortativity = 0.0
        features.append(degree_assortativity)
        times.append(time.time() - start)
        
        start = time.time()
        try:
            degree_pearson_correlation_coefficient = nx.algorithms.assortativity.degree_pearson_correlation_coefficient(g)
            if math.isnan(degree_pearson_correlation_coefficient):
                degree_pearson_correlation_coefficient = 0.0
        except Exception as err:
            traceback.print_exc()
            print("degree_pearson_correlation_coefficient error with  file:", filename)
            degree_pearson_correlation_coefficient = 0.0
        features.append(degree_pearson_correlation_coefficient)
        times.append(time.time() - start)
        

        # add code features here 
        # TO-DO

        # merge into a list
        #print(features)
        return times, torch.LongTensor(features)


    def createGraphFromNXwithTarget(self,g,y,xlen, undirected=True, filename=None):
        """
            Creates a PyTorch Geometric dataset
            from a NetworkX graph
            with node features (called target and represented by y )

            PENDING:
                - append static graph features 
                - append static function code features
        """
        dataset =  self.createGraphFromNX(g,xlen, undirected)
        # 20190906 recent change for later error
        y = torch.LongTensor(y)
        dataset.y = y
        dataset.x_topo_feats, dataset.x_topo_times = self.topological_features(g, filename)
        

        return dataset

    def gnn_mode_on(self):
        """

            remove topo code and document features in memory (not in disk)
            to allow for GNN processing by PyTorch Geometric

        """

        # activate a modifier of the get, that removes those features ONLY in memory
        self.only_graph_features = True

    def gnn_mode_off(self):
        """

            remove topo code and document features in memory (not in disk)
            to allow for GNN processing by PyTorch Geometric

        """

        # activate a modifier of the get, that removes those features ONLY in memory
        self.only_graph_features = False



    def add_code_features(self,data,folder,filename):
        """
            data is the instance that will hold
                - x as a tensor for the node attributes
                - edge_list
                - y the class of the graph
                - x_topo_feats the topological features of the graph
                -> this function will add a series of code features in the form of a vector
                    - concatenates all instructions words into code str
                    - computes numregs and other code features

            filename:
                node.txt file that contains those code_features
        """

        
        with open(folder+filename.replace('edges','nodes'),'r') as f:

            doc = ""
            doc2 = ""
            num_regs = 0
            num_distinct_regs = 0
            num_memaddrs = 0
            #num_imms = 0
            #num_displs = 0
            num_funcs = 0
            num_instrs = 0
            list_regs = []
            list_funcs = []


            # read file
            for line in f.readlines():

                firstsep = line.find("{")
                if firstsep==-1:
                    continue

                attributes = line[firstsep:]
                attr_dict = json.loads(attributes)

                try:
                    

                    # num regs, memaddrs, imms, displs, xrefs from, xrefs to
                    # list of registers, xrefsto,
                    if attr_dict['type'] == 'register' or attr_dict['type'] == 'phrase':
                        num_distinct_regs += 1
                        list_regs.append(attr_dict['raw_content'])
                    elif attr_dict['type'] == 'instr':
                        num_instrs += 1
                        # extract doc from mnemonic_content
                        doc= doc + attr_dict['raw_content'] + " \n"
                        # extract doc2 from raw_content
                        doc2= doc2 + attr_dict['mnemonic_content'] + " \n"
                    elif attr_dict['type'] == 'displacement':
                        num_displs += 1
                        list_regs.append(attr_dict['raw_content'])
                    elif attr_dict['type'] == 'immediate':
                        num_imms += 1
                    elif attr_dict['type'] == 'memory':
                        num_memaddrs += 1
                    elif attr_dict['type'] == 'func':
                        num_funcs += 1
                        list_funcs.append(attr_dict['raw_content'])

                    

                    # apply the BOW model embedding?

                    # pack everything as a vector?
                    # this could be done after the bow model is tested, 
                    # for now 20190728, those values will continue as features for further analysis



                except:
                    pass

        with open(folder+filename,'r') as f:

            
            num_regs = 0
            num_imms = 0
            num_displs = 0
            num_funcs = 0
           


            # read file
            for line in f.readlines():

                firstsep = line.find("{")
                if firstsep==-1:
                    continue

                
                try:
                    
                    attributes = line[firstsep:]
                    attr_dict = json.loads(attributes)


                    # num regs, memaddrs, imms, displs, xrefs from, xrefs to
                    # list of registers, xrefsto,
                    if attr_dict['type'] == 'register' or \
                       attr_dict['type'] == 'phrase' or \
                       attr_dict['type'] == 'displacement':
                        num_regs += 1 
                        #list_regs.append(attr_dict['raw_content'])
                    elif attr_dict['type'] == 'immediate':
                        num_imms += 1
                    elif attr_dict['type'] == 'func':
                        num_funcs += 1
                        
                    

                    # apply the BOW model embedding?

                    # pack everything as a vector?
                    # this could be done after the bow model is tested, 
                    # for now 20190728, those values will continue as features for further analysis



                except Exception as e:
                    #pass
                    print("error with edges.txt ",line,e)

            
            # save everything as a dict for the moment
            code_feats = {
                'nregs': num_regs,
                'num_distinct_regs': num_distinct_regs,
                'ninstrs': num_instrs,
                'ndispls': num_displs,
                'nimms': num_imms,
                'nmaddrs': num_memaddrs,
                'num_funcs': num_funcs,
                'document': doc,
                'document_simplified': doc2,
                'list_regs': list_regs,
                'list_funcs': list_funcs
            }

            data.code_feats= code_feats


        return data





def find_node(fnodes, search_memaddr, search_content, regex):
    """
    find node in the nodes file.
    function to check if graph/code features have been constructed correctly

    """

    fnodes.seek(0)
    #print(hex(search_memaddr), search_content)
    fnodes.seek(0)
    for line in fnodes.readlines():
        tokens2 = line.split('{')
        memaddr_str = tokens2[0]
        try:
            # first find type and select only instr
            j = tokens2[1].find('type": ')
            if j<0:
                print("cannot find type",line)
                nodes_split_errors+=1
                continue
            k = tokens2[1][j:].find(', "content":')
            if k<0:
                print("cannot find type",line)
                nodes_split_errors+=1
                continue
            node_type = tokens2[1][(j+7):(j+k)]
            if node_type != '"instr"':
                continue
            
            j = tokens2[1].find('raw_content":')
            k = tokens2[1][j:].find('"}')
            content_str = tokens2[1][(j+14):(j+k)]
            content_str = content_str.split(';')[0]
            if j<0:
                #print("cannot find content",line)
                nodes_split_errors+=1
                continue
            memaddr = hex(int(memaddr_str))

            current_content = regex.sub(' ',content_str)
            #print("      ",memaddr, content_str)
            if memaddr == hex(search_memaddr) and \
               current_content.strip() == search_content.strip():
                #print("found",memaddr, content_str)
                return True
            elif memaddr == hex(search_memaddr):
                #print("contents is not equal:",search_content+"|"+current_content)
                pass

            #print(memaddr, memaddr_str, content_str)
        except Exception as e:
            print("PROBLEM: ",line, e)

    return False

def check_instructions(assembly_listing_file, nodes_file):
    """
        See test_code_features for explanation.

        goal:
            memory addr and text of instruction are ok

        tasks:
            ok- tranverse nodes_file
            ok- traverse assembly_listing
            ok- transform memaddr from dec to hex 
            - check one to one correspondence of memaddrs in both files
                - the first lines of code are on the same address
                    - sol1) remove them manually , but in xref for example this will happend again
                    - sol2) chekc manually that this is not an error
                    - sol3) implement somth to make sure if the memadrress is repeated on ly the last value remains
            - for each memeaddr, check assembly listing content matches nodes_file content
    """

    # tracking errors
    line_split_errors = 0
    nodes_split_errors = 0
    not_found_instrs = []

    regex = re.compile(r"\s+", re.IGNORECASE)

    with open(assembly_listing_file,'r') as fassembly, open(nodes_file,'r') as fnodes :


        for line in fassembly.readlines()[:-2]:
            tokens2 = line.split(';')
            j = tokens2[0].find(':')
            if j<0:
                print("cannot separate by :",tokens,tokens2)
                line_split_errors+=1
            line = tokens2[0][(j+1):]
            i = line.find(' ')
            if i>-1:
                memaddr = line[:i]
                instr = regex.sub(' ',line[i:])

                int_memaddr = int(memaddr,16)
                
                # debug
                #print(memaddr, str(int_memaddr), instr )

                if not find_node(fnodes, int_memaddr, instr, regex):
                    not_found_instrs.append((int_memaddr,instr))
                else:
                    # make sure thi int_memaddr was not on the list of not founds
                    show_final_results = False
                    if len(not_found_instrs)>0 and \
                       not_found_instrs[-1][0] == int_memaddr:
                        #print("removing previous not found memaddr")
                        #print(not_found_instrs)
                        show_final_results = True
                    while len(not_found_instrs)>0 and \
                       not_found_instrs[-1][0] == int_memaddr:   
                        not_found_instrs.pop()
                    #if show_final_results: print(not_found_instrs)

        #print("\n nodes.txt")
        for line in fnodes.readlines():

            tokens2 = line.split('{')
            memaddr_str = tokens2[0]
            try:
                # first find type and select only instr
                j = tokens2[1].find('type": ')
                if j<0:
                    print("cannot find type",line)
                    nodes_split_errors+=1
                    continue
                k = tokens2[1][j:].find(', "content":')
                if k<0:
                    print("cannot find type",line)
                    nodes_split_errors+=1
                    continue
                node_type = tokens2[1][(j+7):(j+k)]
                if node_type != '"instr"':
                    continue
                
                j = tokens2[1].find('content":')
                k = tokens2[1][j:].find(', "mnemonic_content":')
                content_str = tokens2[1][j:(j+k)]
                if j<0:
                    #print("cannot find content",line)
                    nodes_split_errors+=1
                    continue
                memaddr = hex(int(memaddr_str))
                
                #debug
                #print(memaddr, memaddr_str, content_str)
            except Exception as e:
                print("PROBLEM: ",line, e)
                
    return {
        'line_split_errors': line_split_errors,
        'nodes_split_errors': nodes_split_errors,
        'not_found_instr': not_found_instrs,
    }


def find_instr_in_code(document, search_memaddr, search_content, regex):
    """
            to check for the presence of a search_content in some line in document (code listing)
    """ 
    if search_content.strip() == '':
        return False

    for line in document.split('\n'):
        try:

            content_str = line.split(';')[0]
            current_content = regex.sub(' ',content_str)
            #print("      ",content_str)
            if current_content.strip() == search_content.strip():
                #print("found", content_str)
                return True
            else:
                #print("contents is not equal:",search_content+"|"+current_content)
                pass

            #print(memaddr, memaddr_str, content_str)
        except Exception as e:
            print("PROBLEM: ",line, e)

    return False

def find_registers(instruction_string):
    """
    chec registrers are found in an instruction string
    """

    registers = ["eax","ebx","ecx","edx","esi","edi", "ebp","esp",
                 "rax","rbx","rcx","rdx",
                 "ah","bh","ch","dh","si","di", "bp","sp",
                 "al","bl","cl","dl"]

    found = False
    for reg in registers:
        if instruction_string.lower().find(reg)>-1:
            if instruction_string.lower().find("_"+reg)>-1:
                continue
            found=True 
            break
    return found 


def find_howmany_registers(instruction_string):
    registers = ["eax","ebx","ecx","edx","esi","edi", "ebp","esp",
                 "rax","rbx","rcx","rdx",
                 "ax","bx","cx","dx",
                 "ah","bh","ch","dh","si","di", "bp","sp",
                 "al","bl","cl","dl"]

    found = 0
    positions = []
    for reg in registers:
        i = instruction_string.lower().find(reg)
        if i>-1 and \
           i not in positions and \
           i+1 not in positions and \
           i-1 not in positions and \
           i+2 not in positions and \
           i-2 not in positions and \
           instruction_string.lower().find("_"+reg)==-1 and \
           instruction_string.lower().find("0"+reg)==-1 :
            positions.append(i)
            #print(" found ",reg," in ",instruction_string)
            found+=1 
            
    return found 


def check_document_features(code_feats_dict, nodes_file, assembly_listing_file):
    """
            tracking document(code listing) errors
    """
    line_split_errors = 0
    code_errors = 0
    not_found_instrs = []
    found_instrs = 0
    assembly_listing_code_lines = 0
    num_registers_assembly_listing = 0

    regex = re.compile(r"\s+", re.IGNORECASE)



    with open(assembly_listing_file,'r') as fassembly :

        old_memaddr = -1
        for line in fassembly.readlines()[:-1]:

            
            line_orig = line
            tokens2 = line.split(';')
            j = tokens2[0].find(':')
            if j<0:
                print("cannot separate by :",tokens,tokens2)
                line_split_errors+=1
            line = tokens2[0][(j+1):]
            i = line.find(' ')
            if i>-1:
                memaddr = line[:i]
                instr = regex.sub(' ',line[i:])

                int_memaddr = int(memaddr,16)
                
                # debug
                #print(memaddr, str(int_memaddr), instr )

                if int_memaddr != old_memaddr:
                    assembly_listing_code_lines+=1 
                old_memaddr = int_memaddr

                if find_registers(instr):
                    num_registers_assembly_listing+=find_howmany_registers(instr)

                if not find_instr_in_code(code_feats_dict['document'], int_memaddr, instr, regex):
                    if line_orig.find('; CODE XREF')==-1 and \
                       instr != ' ' and \
                       instr.find(' align ')==-1:
                        not_found_instrs.append((int_memaddr,instr))
                else:

                    found_instrs +=1
                    #print(" found ",memaddr,instr)

                    # make sure thi int_memaddr was not on the list of not founds
                    show_final_results = False
                    if len(not_found_instrs)>0 and \
                       not_found_instrs[-1][0] == int_memaddr:
                        #print("removing previous not found memaddr")
                        #print(not_found_instrs)
                        show_final_results = True
                    while len(not_found_instrs)>0 and \
                       not_found_instrs[-1][0] == int_memaddr:   
                        not_found_instrs.pop()
                        #found_instrs -=1
                    
                        #assembly_listing_code_lines-=1 # approach not working
                    #if show_final_results: print(not_found_instrs)

                
    return {
        'parsing assembly listing line_split_errors': line_split_errors,
        'parsing assembly listing code_errors': code_errors,
        'parsing assembly listing not_found_instr': not_found_instrs,
        'parsing assembly listing found_instrs': found_instrs,
        'assembly listing code lines': assembly_listing_code_lines,
        'assembly listing num regs': num_registers_assembly_listing,
    }



def check_global_code_features(code_feats_dict, nodes_file, assembly_listing_file, result_2):
    """
        code_feats = {
                'nregs': num_regs,
                'ninstrs': num_instrs,
                'ndispls': num_displs,
                'nimms': num_imms,
                'nmaddrs': num_memaddrs,
                'num_funcs': num_funcs,
                'document': doc,
                'document_simplified': doc2,
                'list_regs': list_regs,
                'list_funcs': list_funcs
            }
    """

    regex = re.compile(r"\s+", re.IGNORECASE)

    
    check_num_registers_ok = False

    check_num_instrs_ok = False
    # features num instructions in document code
    doc_num_instrs = result_2['parsing assembly listing found_instrs']
    # features num instructions as a feature 
    feats_num_instrs = code_feats_dict['ninstrs']
    # real num instructions in code
    real_num_instrs = result_2['assembly listing code lines']
    print("checking num instructions: ",
          real_num_instrs,
          doc_num_instrs,
          feats_num_instrs)
    check_num_instrs_ok =  real_num_instrs== doc_num_instrs == feats_num_instrs

    # num regs, displs, imm

                
    print("checking num  registers ",result_2['assembly listing num regs'],
        code_feats_dict['nregs'],
        code_feats_dict['num_distinct_regs'])
    check_num_registers_ok =  result_2['assembly listing num regs'] == code_feats_dict['nregs']     

    

    return {
        'features num instr': feats_num_instrs,
        'features num_instr in document': doc_num_instrs,
        'assembly listing num instr': real_num_instrs,
        '_check_num_instrs_ok': check_num_instrs_ok,

        'assembly listing num regs': result_2['assembly listing num regs'],
        'features num regs': code_feats_dict['nregs'],
        '_check_num_registers_ok': check_num_registers_ok,

        'features num func calls': code_feats_dict['num_funcs'],
        
    }


def check_xrefsto(assembly_listing_file, edges_file, nodes_file):
    """
        ok- get the func edges from edges.txt
        ok- rewrite the nodes.txt with onlly the instructions
        but, putting the link to func under the instructions that calls it 
        ok- manually inspect if it's correct comparing to the assemblly listing 

        
    """

    # tracking errors
    line_split_errors = 0
    nodes_split_errors = 0
    not_found_instrs = []
    xrefs_to_instrs = []

    regex = re.compile(r"\s+", re.IGNORECASE)

    funcs_list = []
    with open(nodes_file,'r') as fnodes, open(edges_file,'r') as fedges :


        fedges.seek(0)
        for line in fedges.readlines():
            tokens2 = line.split('{')
            memaddr_str = tokens2[0]
            try:
                # first find type and select only instr
                j = tokens2[1].find('type": ')
                if j<0:
                    print("cannot find type",line)
                    nodes_split_errors+=1
                    continue
                k = tokens2[1][j:].find(' }')
                if k<0:
                    print("cannot find type",line)
                    nodes_split_errors+=1
                    continue
                node_type = tokens2[1][(j+7):(j+k)]
                if node_type == '"func"':
                    splits = line.split(' ')
                    m1 = splits[0]
                    m2 = splits[1]
                    funcs_list.append((hex(int(m1)),hex(int(m2)))  )
            except Exception as e:
                print("PROBLEM: ",line, e)

        xrefsto_origins = [e[0] for e in funcs_list]
        xrefsto_dests = [e[1] for e in funcs_list]

        fnodes.seek(0)
        for line in fnodes.readlines():
            
            tokens2 = line.split('{')
            memaddr_str = tokens2[0]
            try:
                # first find type and select only instr
                j = tokens2[1].find('type": ')
                if j<0:
                    print("cannot find type",line)
                    nodes_split_errors+=1
                    continue
                k = tokens2[1][j:].find(', "content":')
                if k<0:
                    print("cannot find type",line)
                    nodes_split_errors+=1
                    continue
                node_type = tokens2[1][(j+7):(j+k)]
                if node_type != '"instr"':
                    continue
                
                
                memaddr = hex(int(memaddr_str))

                # DEBUG print  xrefstTO and instructions
                if memaddr in xrefsto_origins:
                    l = xrefsto_origins.index(memaddr)
                    #print(memaddr,line)
                    #print("        xreft to ",xrefsto_dests[l])
                    xrefs_to_instrs.append((line, xrefsto_dests[l]))
                # printf bug verification
                #elif memaddr == hex(int("0045E3FA",16)):
                #    print(memaddr, line)

            except Exception as e:
                print("PROBLEM: ",line, e)

    #pprint(funcs_list)
                    
    return {
        'num funcs calls from edges and nodes files': len(funcs_list),
        
        'funcs calls xrefsto info': xrefs_to_instrs,
        'funcs calls list': funcs_list
    }




def test_code_features(code_feats_dict, assembly_listing_file, nodes_file, edges_file):

    """
        params
            - code-feats_dict is a python dictionary with the code features of an assembly function.
            - assembly_listing is a txt file with the listing of assemblloy instructions directly extracted from the disassembler(IDA)
            - nodes_file is the generated nodes.txt file containing all the nodes of the graph of the function(each node is an instruction or a register or a memaddr or..)
            - edges_file is the generated edges.txt file which contains all the connections between nodes (registers to the instructions they are called in or used in, instructions that call to other instructions or even the sequential order of instructions)

        This mega function will check the following conditions
            1) memory addr and text of instruction are ok
            2) all instructions are there
            3) Xrefrom are correct
                ; CODE XREF: ... 
                vs
                Xrefs in the list
    
            4) num funcs check also 
                after Call without short_loc or loc

            5) the edges are correct.. this one is pretty difficult, for the moment it will have to happend visually
        
        Return:
            prints kind of a report of what controls passed and what controls failed

    """

    result_1 = check_instructions(assembly_listing_file, nodes_file)

    

    result_2 = check_document_features(code_feats_dict, nodes_file, assembly_listing_file)


    result_3 = check_global_code_features(code_feats_dict, nodes_file, assembly_listing_file, result_2)



    result_4 = check_xrefsto(assembly_listing_file, edges_file, nodes_file)

    result_1.update(result_2)
    result_1.update(result_3)
    result_1.update(result_4)
    pprint(result_1)



def add_node_degree(dataset_folder):
    print("loading", dataset_folder)
    dataset = FunctionsDataset(root=dataset_folder)
    print("num samples:",len(dataset))
    #print("num classes:",dataset.num_classes)
    print("num features:",dataset.num_features)


    max_degree = 0
    for j in range(len(dataset)):

        # if j > 2:
        #     break
        data = dataset[j]
        
        # compute node degrees
        elist1 = data.edge_index.tolist()[0]
        elist2 = data.edge_index.tolist()[1]
        edgelist = []
        for k in range(len(elist1)):
            edgelist.append((elist1[k],elist2[k]))
        # just count how many times a value appears in one list = degree
        g = nx.Graph(edgelist)


        # get all node ids
        node_ids = list(set(elist1))

        # degrees
        degrees = nx.degree(g)
        # correct for unconnected nodes
        node_ids = range(data.x.shape[0])
        degrees_dict = dict(degrees)
        for idx in node_ids:
            if idx not in degrees_dict.keys():
                degrees_dict[idx]=0

        # transform to torch
        degrees = torch.FloatTensor(list(degrees_dict.values()))
        degrees = degrees.view(data.x.shape[0],-1)

        # save max degree
        current_max = max(list(degrees_dict.values()))
        if max_degree < current_max:
            max_degree = current_max


    # redo with max_degree
    T.OneHotDegree(max_degree)
        
    for j in range(len(dataset)):

        # if j > 2:
        #     break
        data = dataset[j]
        print("data.x")
        pprint(data.x)
        print("edge list", data.edge_index)


        # compute node degrees
        print("edge_list origins:")
        pprint(data.edge_index.tolist()[0])
        print("edge_list destinations:")
        pprint(data.edge_index.tolist()[1])
        elist1 = data.edge_index.tolist()[0]
        elist2 = data.edge_index.tolist()[1]
        edgelist = []
        for k in range(len(elist1)):
            edgelist.append((elist1[k],elist2[k]))
        # just count how many times a value appears in one list = degree
        g = nx.Graph(edgelist)



        # get all node ids
        node_ids = list(set(elist1))
        print("node_ids",node_ids)        
        print("node_ids from nx", g.nodes())

        # degrees
        degrees = nx.degree(g)
        print("degrees")
        pprint(degrees)
        pprint(g.degree())

        # correct for unconnected nodes
        node_ids = range(data.x.shape[0])
        degrees_dict = dict(degrees)
        for idx in node_ids:
            if idx not in degrees_dict.keys():
                degrees_dict[idx]=0

        print("node_ids",node_ids )
        print("degrees keys", degrees_dict.keys())
        print("degrees keys", degrees_dict.values(),"are they sorted?")




        # transform to torch
        print("degrees values()")

        degrees = torch.FloatTensor(list(degrees_dict.values()))
        pprint(degrees)
        degrees = degrees.view(data.x.shape[0],-1)

        # save max degree
        current_max = max(list(degrees_dict.values()))
        if max_degree < current_max:
            max_degree = current_max


        # add new column to x
        data.x = torch.cat((data.x, degrees), dim=1)
        print("transformed data.x")
        pprint(data.x)

        # save graph
        dataset.save_changes(j,data)




def add_node_degree_v2(dataset_folder):
    """
        follows PyG/benchmark/kernel/datasets.py indications
        to use a T.OneHotDegree as a transform for the dataset

    """
    print("loading", dataset_folder)
    dataset = FunctionsDataset(root=dataset_folder)
    print("num samples:",len(dataset))
    #print("num classes:",dataset.num_classes)
    print("num features:",dataset.num_features)


    max_degree = 0
    degs = []
    for data in dataset:
        degs += [degree(data.edge_index[0], dtype=torch.long)]
        max_degree = max(max_degree, degs[-1].max().item())

    if max_degree < 2000:
        dataset.transform = T.OneHotDegree(max_degree)
    else:
        deg = torch.cat(degs, dim=0).to(torch.float)
        mean, std = deg.mean().item(), deg.std().item()
        dataset.transform = NormalizedDegree(mean, std)

    return dataset


def add_node_degree_v3(dataset_folder):
    dataset = add_node_degree_v2(dataset_folder)
    for j in range(len( dataset)):
        d = dataset[j]
        pprint(d.x)
        dataset.save_changes(j,d)


def switch_dataset_class(cl_origin=0.0, cl_dest=1.0, root=''):
    """
        During training of v1, 
        classes 0 and 9 have very low number of samples.
        to balance dataset those clases are removed (samples of that classs are not taken into account)

        but since mlps need the classes to be in order 0 to n-1, 
        we need to mode class 0 to 8 , so in the end we remove examples of classes 8 and 9 only.

    """
    dataset = FunctionsDataset(root=root)

    # for every sample in the dataset

    for j in range(len(dataset)):
        data = dataset[j]
    
        # if cl = cl_origin then change it to cl_dest
        if data.y == int(cl_origin):
            #print(" found graph ",j," with data.y=",data.y)
            data.y = int(cl_dest)

        # save changes to disk
        dataset.save_changes(j,data)
    

def combine_bow_features_with_graph(cl_origin=0.0, cl_dest=1.0, root=''):
    """
        During training of v1, 
        classes 0 and 9 have very low number of samples.
        to balance dataset those clases are removed (samples of that classs are not taken into account)

        but since mlps need the classes to be in order 0 to n-1, 
        we need to mode class 0 to 8 , so in the end we remove examples of classes 8 and 9 only.

    """
    dataset = FunctionsDataset(root=root)

    # for every sample in the dataset

    for j in range(len(dataset)):
        data = dataset[j]
    
        # add the corresponding row of the precomputed tfidf bow
        # match symbols_dataset_3_precomp_splits_undersample_max2/training_set/processed 
        # with
        # symbols_dataset_3_precomp_splits_undersample_max2/X_train.pickle
        # and 
        # symbols_dataset_3_precomp_splits_undersample_max2/test_set/processed 
        # with
        # symbols_dataset_3_precomp_splits_undersample_max2/X_test.pickle

        # extract the corresponding row for X_train or X_test...



        # find column order!!!
        #   topos = dataset[j].__getattribute__('x_topo_feats')

        #   # get all the features
        #   cfeats  =[d['document'],d['document_simplified'],d['list_funcs'], d['nregs'], d['num_distinct_regs'], d['ninstrs'], d['ndispls'], d['nimms'], d['nmaddrs'], d['num_funcs'],]
        #       cfeats.extend(topos)

        # then where is the embedding???? X_train_tfidf_document.pickle!
        # is it in the first column like same order ? or is it somewhere else?

        # find the row number is equivalent to the dataset value?


        # save changes to disk
        dataset.save_changes(j,data)
    





if __name__=='__main__':


    add_node_degree_v3('./tmp/symbols_dataset_1')