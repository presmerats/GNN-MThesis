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
from pprint import pprint
import numpy as np
from numpy.random import choice
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from random import shuffle

import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader


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
    # read nodes, edges and  edge attributes
    g = nx.read_edgelist(folder+filename)
    # read node attributes
    with open(folder+filename.replace('edges','nodes'),'r') as f:
        for line in f.readlines():
            firstsep = line.find("{")
            if firstsep==-1:
                continue

            tokens = line[firstsep:]
            nodeid = line[:firstsep]
            
            attr_dict = json.loads(tokens)
            
            if nodeid not in g.node.keys():
                g.add_node(nodeid)
            
            for k,v in attr_dict.items():
                try:
                    g.node[nodeid][k] = v
                except:
                    print("problem with ",nodeid,attr_dict)
                    print("{} in g.node.keys() ?".format(nodeid),nodeid in g.node.keys())
                
    return g

def prepareLabelDict(g, attr):
    label_dict = {}
    for node in g.nodes():
        if attr == 'all':
            label_dict[node]=node + " " + \
                             g.node[node]['content']
        else:
            label_dict[node]=g.node[node][attr]
    return label_dict

def plotGraph(g, label=None):
    
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
    g = readGraph(folder, filename)
    plotGraph(g,label)
        
class FunctionsDataset(Dataset):
    node_translation = {}  # translates from nodeid like im34,r2 to an autoincremented integer
    nodeidmax = -1
    instr_types = ['instr','register','immediate','memory',
        'displacement','phrase','func','unknown']
    x86_instr_set = [ i.lower() for i in ['AAA', 'AAD', 'AAM', 'AAS', 'ADC', 'ADCX', 'ADD', 
                     'ADDPD', 'ADDPS', 'ADDSD', 'ADDSS', 'ADDSUBPD', 
                     'ADDSUBPS', 'ADOX', 'AESDEC', 'AESDECLAST', 'AESENC',
                     'AESENCLAST', 'AESIMC', 'AESKEYGENASSIST', 'AND', 'ANDN', 
                     'ANDNPD', 'ANDNPS', 'ANDPD', 'ANDPS', 'ARPL', 'BEXTR', 
                     'BLENDPD', 'BLENDPS', 'BLENDVPD', 'BLENDVPS', 'BLSI', 
                     'BLSMSK', 'BLSR', 'BNDCL', 'BNDCN', 'BNDCU', 'BNDLDX', 
                     'BNDMK', 'BNDMOV', 'BNDSTX', 'BOUND', 'BSF', 'BSR', 'BSWAP', 'BT', 'BTC', 'BTR', 'BTS', 'BZHI', 'CALL', 'CBW', 'CDQ', 'CDQE', 'CLAC', 'CLC', 'CLD', 'CLDEMOTE', 'CLFLUSH', 'CLFLUSHOPT', 'CLI', 'CLTS', 'CLWB', 'CMC', 'CMOVcc', 'CMP', 'CMPPD', 'CMPPS', 'CMPS', 'CMPSB', 'CMPSD', 'CMPSQ', 'CMPSS', 'CMPSW', 'CMPXCHG', 'COMISD', 'COMISS', 'CPUID', 'CQO', 'CWD', 'CWDE', 'DAA', 'DAS', 'DEC', 'DIV', 'DIVPD', 'DIVPS', 'DIVSD', 'DIVSS', 'DPPD', 'DPPS', 'EMMS', 'ENTER', 'EXTRACTPS', 'FABS', 'FADD', 'FADDP', 'FBLD', 'FBSTP', 'FCHS', 'FCLEX', 'FCMOVcc', 'FCOM', 'FCOMI', 'FCOMIP', 'FCOMP', 'FCOMPP', 'FCOS', 'FDECSTP', 'FDIV', 'FDIVP', 'FDIVR', 'FDIVRP', 'FFREE', 'FIADD', 'FICOM', 'FICOMP', 'FIDIV', 'FIDIVR', 'FILD', 'FIMUL', 'FINCSTP', 'FINIT', 'FIST', 'FISTP', 'FISTTP', 'FISUB', 'FISUBR', 'FLD', 'FLDCW', 'FLDENV', 'FLDPI', 'FLDZ', 'FMUL', 'FMULP', 'FNCLEX', 'FNINIT', 'FNOP', 'FNSAVE', 'FNSTCW', 'FNSTENV', 'FNSTSW', 'FPATAN', 'FPREM', 'FPTAN', 'FRNDINT', 'FRSTOR', 'FSAVE', 'FSCALE', 'FSIN', 'FSINCOS', 'FSQRT', 'FST', 'FSTCW', 'FSTENV', 'FSTP', 'FSTSW', 'FSUB', 'FSUBP', 'FSUBR', 'FSUBRP', 'FTST', 'FUCOM', 'FUCOMI', 'FUCOMIP', 'FUCOMP', 'FUCOMPP', 'FWAIT', 'FXAM', 'FXCH', 'FXRSTOR', 'FXSAVE', 'FXTRACT', 'HADDPD', 'HADDPS', 'HLT', 'HSUBPD', 'HSUBPS', 'IDIV', 'IMUL', 'IN', 'INC', 'INS', 'INSB', 'INSD', 'INSERTPS', 'INSW', 'INTO', 'INVD', 'INVLPG', 'INVPCID', 'IRET', 'IRETD', 'JMP', 'Jcc', 'KADDB', 'KADDD', 'KADDQ', 'KADDW', 'KANDB', 'KANDD', 'KANDNB', 'KANDND', 'KANDNQ', 'KANDNW', 'KANDQ', 'KANDW', 'KMOVB', 'KMOVD', 'KMOVQ', 'KMOVW', 'KNOTB', 'KNOTD', 'KNOTQ', 'KNOTW', 'KORB', 'KORD', 'KORQ', 'KORTESTB', 'KORTESTD', 'KORTESTQ', 'KORTESTW', 'KORW', 'KSHIFTLB', 'KSHIFTLD', 'KSHIFTLQ', 'KSHIFTLW', 'KSHIFTRB', 'KSHIFTRD', 'KSHIFTRQ', 'KSHIFTRW', 'KTESTB', 'KTESTD', 'KTESTQ', 'KTESTW', 'KUNPCKBW', 'KUNPCKDQ', 'KUNPCKWD', 'KXNORB', 'KXNORD', 'KXNORQ', 'KXNORW', 'KXORB', 'KXORD', 'KXORQ', 'KXORW', 'LAHF', 'LAR', 'LDDQU', 'LDMXCSR', 'LDS', 'LEA', 'LEAVE', 'LES', 'LFENCE', 'LFS', 'LGDT', 'LGS', 'LIDT', 'LLDT', 'LMSW', 'LOCK', 'LODS', 'LODSB', 'LODSD', 'LODSQ', 'LODSW', 'LOOP', 'LOOPcc', 'LSL', 'LSS', 'LTR', 'LZCNT', 'MASKMOVDQU', 'MASKMOVQ', 'MAXPD', 'MAXPS', 'MAXSD', 'MAXSS', 'MFENCE', 'MINPD', 'MINPS', 'MINSD', 'MINSS', 'MONITOR', 'MOV', 'MOVAPD', 'MOVAPS', 'MOVBE', 'MOVD', 'MOVDDUP', 'MOVDIRI', 'MOVDQA', 'MOVDQU', 'MOVHLPS', 'MOVHPD', 'MOVHPS', 'MOVLHPS', 'MOVLPD', 'MOVLPS', 'MOVMSKPD', 'MOVMSKPS', 'MOVNTDQ', 'MOVNTDQA', 'MOVNTI', 'MOVNTPD', 'MOVNTPS', 'MOVNTQ', 'MOVQ', 'MOVS', 'MOVSB', 'MOVSD', 'MOVSHDUP', 'MOVSLDUP', 'MOVSQ', 'MOVSS', 'MOVSW', 'MOVSX', 'MOVSXD', 'MOVUPD', 'MOVUPS', 'MOVZX', 'MPSADBW', 'MUL', 'MULPD', 'MULPS', 'MULSD', 'MULSS', 'MULX', 'MWAIT', 'NEG', 'NOP', 'NOT', 'OR', 'ORPD', 'ORPS', 'OUT', 'OUTS', 'OUTSB', 'OUTSD', 'OUTSW', 'PABSB', 'PABSD', 'PABSQ', 'PABSW', 'PACKSSDW', 'PACKSSWB', 'PACKUSDW', 'PACKUSWB', 'PADDB', 'PADDD', 'PADDQ', 'PADDSB', 'PADDSW', 'PADDUSB', 'PADDUSW', 'PADDW', 'PALIGNR', 'PAND', 'PANDN', 'PAUSE', 'PAVGB', 'PAVGW', 'PBLENDVB', 'PBLENDW', 'PCLMULQDQ', 'PCMPEQB', 'PCMPEQD', 'PCMPEQQ', 'PCMPEQW', 'PCMPESTRI', 'PCMPESTRM', 'PCMPGTB', 'PCMPGTD', 'PCMPGTQ', 'PCMPGTW', 'PCMPISTRI', 'PCMPISTRM', 'PDEP', 'PEXT', 'PEXTRB', 'PEXTRD', 'PEXTRQ', 'PEXTRW', 'PHADDD', 'PHADDSW', 'PHADDW', 'PHMINPOSUW', 'PHSUBD', 'PHSUBSW', 'PHSUBW', 'PINSRB', 'PINSRD', 'PINSRQ', 'PINSRW', 'PMADDUBSW', 'PMADDWD', 'PMAXSB', 'PMAXSD', 'PMAXSQ', 'PMAXSW', 'PMAXUB', 'PMAXUD', 'PMAXUQ', 'PMAXUW', 'PMINSB', 'PMINSD', 'PMINSQ', 'PMINSW', 'PMINUB', 'PMINUD', 'PMINUQ', 'PMINUW', 'PMOVMSKB', 'PMOVSX', 'PMOVZX', 'PMULDQ', 'PMULHRSW', 'PMULHUW', 'PMULHW', 'PMULLD', 'PMULLQ', 'PMULLW', 'PMULUDQ', 'POP', 'POPA', 'POPAD', 'POPCNT', 'POPF', 'POPFD', 'POPFQ', 'POR', 'PREFETCHW', 'PREFETCHh', 'PSADBW', 'PSHUFB', 'PSHUFD', 'PSHUFHW', 'PSHUFLW', 'PSHUFW', 'PSIGNB', 'PSIGND', 'PSIGNW', 'PSLLD', 'PSLLDQ', 'PSLLQ', 'PSLLW', 'PSRAD', 'PSRAQ', 'PSRAW', 'PSRLD', 'PSRLDQ', 'PSRLQ', 'PSRLW', 'PSUBB', 'PSUBD', 'PSUBQ', 'PSUBSB', 'PSUBSW', 'PSUBUSB', 'PSUBUSW', 'PSUBW', 'PTEST', 'PTWRITE', 'PUNPCKHBW', 'PUNPCKHDQ', 'PUNPCKHQDQ', 'PUNPCKHWD', 'PUNPCKLBW', 'PUNPCKLDQ', 'PUNPCKLQDQ', 'PUNPCKLWD', 'PUSH', 'PUSHA', 'PUSHAD', 'PUSHF', 'PUSHFD', 'PUSHFQ', 'PXOR', 'RCL', 'RCPPS', 'RCPSS', 'RCR', 'RDFSBASE', 'RDGSBASE', 'RDMSR', 'RDPID', 'RDPKRU', 'RDPMC', 'RDRAND', 'RDSEED', 'RDTSC', 'RDTSCP', 'REP', 'REPE', 'REPNE', 'REPNZ', 'REPZ', 'RET', 'ROL', 'ROR', 'RORX', 'ROUNDPD', 'ROUNDPS', 'ROUNDSD', 'ROUNDSS', 'RSM', 'RSQRTPS', 'RSQRTSS', 'SAHF', 'SAL', 'SAR', 'SARX', 'SBB', 'SCAS', 'SCASB', 'SCASD', 'SCASW', 'SETcc', 'SFENCE', 'SGDT', 'SHL', 'SHLD', 'SHLX', 'SHR', 'SHRD', 'SHRX', 'SHUFPD', 'SHUFPS', 'SIDT', 'SLDT', 'SMSW', 'SQRTPD', 'SQRTPS', 'SQRTSD', 'SQRTSS', 'STAC', 'STC', 'STD', 'STI', 'STMXCSR', 'STOS', 'STOSB', 'STOSD', 'STOSQ', 'STOSW', 'STR', 'SUB', 'SUBPD', 'SUBPS', 'SUBSD', 'SUBSS', 'SWAPGS', 'SYSCALL', 'SYSENTER', 'SYSEXIT', 'SYSRET', 'TEST', 'TPAUSE', 'TZCNT', 'UCOMISD', 'UCOMISS', 'UD', 'UMONITOR', 'UMWAIT', 'UNPCKHPD', 'UNPCKHPS', 'UNPCKLPD', 'UNPCKLPS', 'VALIGND', 'VALIGNQ', 'VBLENDMPD', 'VBLENDMPS', 'VBROADCAST', 'VCOMPRESSPD', 'VCOMPRESSPS', 'VDBPSADBW', 'VERR', 'VERW', 'VEXPANDPD', 'VEXPANDPS', 'VFIXUPIMMPD', 'VFIXUPIMMPS', 'VFIXUPIMMSD', 'VFIXUPIMMSS', 'VFPCLASSPD', 'VFPCLASSPS', 'VFPCLASSSD', 'VFPCLASSSS', 'VGATHERDPD', 'VGATHERDPS', 'VGATHERQPD', 'VGATHERQPS', 'VGETEXPPD', 'VGETEXPPS', 'VGETEXPSD', 'VGETEXPSS', 'VGETMANTPD', 'VGETMANTPS', 'VGETMANTSD', 'VGETMANTSS', 'VMASKMOV', 'VPBLENDD', 'VPBLENDMB', 'VPBLENDMD', 'VPBLENDMQ', 'VPBLENDMW', 'VPBROADCAST', 'VPBROADCASTB', 'VPBROADCASTD', 'VPBROADCASTM', 'VPBROADCASTQ', 'VPBROADCASTW', 'VPCMPB', 'VPCMPD', 'VPCMPQ', 'VPCMPUB', 'VPCMPUD', 'VPCMPUQ', 'VPCMPUW', 'VPCMPW', 'VPCOMPRESSD', 'VPCOMPRESSQ', 'VPCONFLICTD', 'VPCONFLICTQ', 'VPERMB', 'VPERMD', 'VPERMILPD', 'VPERMILPS', 'VPERMPD', 'VPERMPS', 'VPERMQ', 'VPERMW', 'VPEXPANDD', 'VPEXPANDQ', 'VPGATHERDD', 'VPGATHERDQ', 'VPGATHERQD', 'VPGATHERQQ', 'VPLZCNTD', 'VPLZCNTQ', 'VPMASKMOV', 'VPMOVDB', 'VPMOVDW', 'VPMOVQB', 'VPMOVQD', 'VPMOVQW', 'VPMOVSDB', 'VPMOVSDW', 'VPMOVSQB', 'VPMOVSQD', 'VPMOVSQW', 'VPMOVSWB', 'VPMOVUSDB', 'VPMOVUSDW', 'VPMOVUSQB', 'VPMOVUSQD', 'VPMOVUSQW', 'VPMOVUSWB', 'VPMOVWB', 'VPMULTISHIFTQB', 'VPROLD', 'VPROLQ', 'VPROLVD', 'VPROLVQ', 'VPRORD', 'VPRORQ', 'VPRORVD', 'VPRORVQ', 'VPSCATTERDD', 'VPSCATTERDQ', 'VPSCATTERQD', 'VPSCATTERQQ', 'VPSLLVD', 'VPSLLVQ', 'VPSLLVW', 'VPSRAVD', 'VPSRAVQ', 'VPSRAVW', 'VPSRLVD', 'VPSRLVQ', 'VPSRLVW', 'VPTERNLOGD', 'VPTERNLOGQ', 'VPTESTMB', 'VPTESTMD', 'VPTESTMQ', 'VPTESTMW', 'VPTESTNMB', 'VPTESTNMD', 'VPTESTNMQ', 'VPTESTNMW', 'VRANGEPD', 'VRANGEPS', 'VRANGESD', 'VRANGESS', 'VREDUCEPD', 'VREDUCEPS', 'VREDUCESD', 'VREDUCESS', 'VRNDSCALEPD', 'VRNDSCALEPS', 'VRNDSCALESD', 'VRNDSCALESS', 'VSCALEFPD', 'VSCALEFPS', 'VSCALEFSD', 'VSCALEFSS', 'VSCATTERDPD', 'VSCATTERDPS', 'VSCATTERQPD', 'VSCATTERQPS', 'VTESTPD', 'VTESTPS', 'VZEROALL', 'VZEROUPPER', 'WAIT', 'WBINVD', 'WRFSBASE', 'WRGSBASE', 'WRMSR', 'WRPKRU', 'XABORT', 'XACQUIRE', 'XADD', 'XBEGIN', 'XCHG', 'XEND', 'XGETBV', 'XLAT', 'XLATB', 'XOR', 'XORPD', 'XORPS', 'XRELEASE', 'XRSTOR', 'XRSTORS', 'XSAVE', 'XSAVEC', 'XSAVEOPT', 'XSAVES', 'XSETBV', 'XTEST', 'EACCEPT', 'EACCEPTCOPY', 'EADD', 'EAUG', 'EBLOCK', 'ECREATE', 'EDBGRD', 'EDBGWR', 'EDECVIRTCHILD', 'EENTER', 'EEXIT', 'EEXTEND', 'EGETKEY', 'EINCVIRTCHILD', 'EINIT', 'ELBUC', 'ELDB', 'ELDBC', 'ELDU', 'EMODPE', 'EMODPR', 'EMODT', 'ENCLS', 'ENCLU', 'ENCLV', 'EPA', 'ERDINFO', 'EREMOVE', 'EREPORT', 'ERESUME', 'ESETCONTEXT', 'ETRACK', 'ETRACKC', 'EWB', 'INVEPT', 'INVVPID', 'VMCALL', 'VMCLEAR', 'VMFUNC', 'VMLAUNCH', 'VMPTRLD', 'VMPTRST', 'VMREAD', 
                     'VMRESUME', 'VMWRITE', 'VMXOFF', 'VMXON']]
    
    myprocessed_filenames = []
    problematic_files = []
    my_classes = set([])
    
    def __init__(self, root, transform=None, pre_transform=None):
        super(FunctionsDataset, self).__init__(root, transform, pre_transform)
        print(" end of __init__()")

    @property
    def raw_file_names(self):
        return ['graphs01']

    @property
    def processed_file_names(self):
        #return self.myprocessed_filenames
        # update the list of processed filenames
        # assumes base dir?? how to resolve that?
        
        if len(self.myprocessed_filenames)==0:

            # if myprocessed_filenames is not initialized,
            # read all filenames from processed folder

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
        k = len(self.my_classes)
        if k <= 0 or True:
            k=0
            # read the num classes by the subdirs in raw/graphs01
            rawfolder = os.path.join(self.root, 'raw/graphs01')
            for item in os.listdir(rawfolder):
                if os.path.isdir(os.path.join(rawfolder, item)):
                    k+=1
        return k 
    
    @property
    def num_features(self):
        # this should be dynamic!!
        # load one of the processed files and count the features in there!
        graph0 = self.get(0)
        return graph0.x.shape[1]
        #return 4
    
    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        # Download to `self.raw_dir`.
        # this step is not implemented. You need to manually
        # copy the folders that contain the output of the plugin
        # inside the raw folder
        print("Not implemented. Missing folders with graph files in txt for Nx format, for each program to be included in the dataset")
        

    def process(self):
        """
          this function must read the datasets
          every graph must create a file and increment an index
          and also save the filename to self.myprocessed_files
        """
        
        # Read data into huge `Data` list.
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


        
        print()
        print("Finished reading-processing dataset,")
        print("processed: {} total files, {} created graph files, {} problematic files".format(
            i+len(self.problematic_files),i,len(self.problematic_files)))
        print()


    def shuffle(self):
        # randomize the list of processed files form disk
        #shuffle(self.processed_file_names)
        return self
         
    def get(self, idx):
        # idx is a torch.LongTensor
        # extract a list from the longTensor
        # load each file separately --
        # load then in the final data tensor
        pointers = []
        if isinstance(idx, int):
            # transform this into filename from a list of processed paths
            
            try:
                #realfile = self.processed_paths[idx]
                realfile = self.myprocessed_filenames[idx]
                filename = os.path.join(self.processed_dir,realfile)
                data = torch.load(filename)
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
        try:
            y = self.extractClassLabelFromFolderName(os.path.dirname(filename))
            g = readGraph('',filename)
            if not self.verify_created_graph(g):
                print("problem with: ", filename, len(g.nodes()), len(g.edges()))
                return None
            xlen = len(g.nodes())
            data = self.createGraphFromNXwithTarget(g,y,xlen,undirected=True)
            
            return data
        except Exception as err:
            #print(filename)
            traceback.print_exc()
            return None
        
    
    def extractClassLabelFromFolderName(self, root):
        # extract class from folder name
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
            print("Unknown class! ",root)


        
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
        self.nodeidmax = -1
        self.node_translation = {}
        
    def translateNodeIds(self,nodeid,g):
        """
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
            
            version2: return index_type?
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
        
        if node_type not in edge_list_1.keys():
            edge_list_1[node_type]=[n1]
            
        else:
            edge_list_1[node_type].append(n1)
            
        return edge_list_1
    
    def merge_edge_list(self, edge_list):
        """
            pre: edge_list a dict of lists of ints
        """
        result_list = []
        previous_max_id = 0
        for k,vlist in edge_list.items():
            result_list.extend([ elemid + previous_max_id for elemid in vlist])
            previous_max_id = max(vlist) + previous_max_id
            
        return result_list
    
    def createGraphFromNX(self, g, xlen, undirected=True):
        """
            Creates a PyTorch Geometric dataset
            from a NetworkX graph

            node features -> to one-hot-encoding
                - type (string)
                - content
                    - reg number string -> to int
                    - displacement number string -> to int?
                    - memory/ other string -> to what?

        """
        # get edge list
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

        # features
        # use e[0] and e[1] original memoray address values as a feature
        # add the rest of the features: type, content, ...
        
        # starting point: 
        #    g.node[node]={
        #      'type': 'register' ,
        #       'content' : integer}
        #   and node as a memaddr
        # result:
        #  x [ type_as_one-hot-enc, int(content), int(node)]
        n = xlen
        x = {}
        for node in g.nodes():
            #  all attr of a node are converted to int

            #print(node, g.node[node])
            nodeid = self.translateNodeIds(node, g)
            
            # int
            # node is an int or a string  im32 -> how to separate
            # if it contains any alphabetical number -> no memaddres
            memaddr = -1.0
            if not re.search('[^0-9]', node):
                memaddr = int(node)
                

            # vector with 0's and a 1
            # version2: replaced with int
            instr_type = self.one_hot_nodetype_v2(g.node[node])
            
            # int
            if 'type' in g.node[node].keys() and \
               g.node[node]['type'] != 'instr':
                try:
                    content_not_instr = int(g.node[node]['content'])
                except:
                    content_not_instr = 0
                
            else:
                content_not_instr = 0
                
            # vector with 0's and a 1 in the actual instruction
            # version2: replaced with int
            content_instr = self.one_hot_instruction_v2(g.node[node])
            
            
            x[nodeid] = [memaddr, content_not_instr]
            #print("content_instr: ",content_instr)
            #x[nodeid].extend(content_instr) # one_hot_nodetype()
            x[nodeid].append(content_instr) # one_hot_nodetype_v2()
            #print("instr_type: ", instr_type)
            #x[nodeid].extend(instr_type) # one_hot_instruction()
            x[nodeid].append(instr_type) # one_hot_instruciton_v2()
            # total 4 features per node
            
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
    
        
        return self.createGraph(x, edge_index)

    def createGraphFromNXwithTarget(self,g,y,xlen, undirected=True):
        """
            Creates a PyTorch Geometric dataset
            from a NetworkX graph
            with node features (called target and represented by y )

            PENDING:
                - appepnd many Datas to the dataset
        """
        dataset =  self.createGraphFromNX(g,xlen, undirected)
        y = torch.LongTensor(y)
        dataset.y = y 
        return dataset
    
   