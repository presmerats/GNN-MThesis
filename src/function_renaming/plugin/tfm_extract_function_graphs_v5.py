"""
	Goal:
		write a series of txt files to later be imported from Networkx library.
		
	structure of the files:
		1) Nodes:  mem-address = id?
				   features: 
					- type: instr, data_addr, register, immediate, funccall
					- contents: 
						- if instr  then string in a bagofwords-one-of-k-coding
						- if data_addr then ?
						- if register then name and address -> each transformed to int id
						- if immediate then value -> transformed to int id..?
						- if funccall -> funcaddr as int id
		2) edges: id auto increment integer
				 id of mem addr origin and id of mem addr destination
				 Directed edges
				 -> no attributes for the moment..
				 -> possible attributes:
					- type: to register, to data, to funccall, to immediate, to code
	

	REFACTOR TASKS:
		- file ini in a function
		- organize node and edge saving with a single method for each (according to networkx or PyG)
			- PYG format
				- node feature file: node id, feat1, feat2, ..
				- edge feature file: edge id, feat1, feat2, feat3,...
				- edge_index file: source node, destination node (later must be converted
			- Networkx format:
				- edge list: 1 2 ('weight':7, 'color': green, ...)
					- direct import into Networkx
				- node features file:  node id { 'feat1_name': feat1_val, ...}
					- manually add those features into the nodes once imported from edge_list
"""
from idaapi import *
import os
import datetime
import string


class codeNode():
	def __init__(self, memaddr, type, content, fd_nodes=None, fd_edges=None):
		
		
		self.memaddr = memaddr
		self.type = type 
		self.content = content
		self.mnemonic_content = ''
		self.raw_content = ''
		self.fd_nodes = fd_nodes
		self.fd_edges = fd_edges
		#Message("init: %s  %s  %s\n" % (self.memaddr, self.type, self.content))
	
	def saveNode(self, results_dict):
		global autoid
		global autoid_phrase
		#Message("before save:  %s  %s  %s\n" % (self.memaddr, self.type, self.content))
		
		self.sanitize_content()
		
		if self.memaddr == '0' and self.type == 'register':
			self.memaddr = 'r' + self.content
		elif self.memaddr == '0' and self.type == 'immediate':
			self.memaddr = 'im' + self.content
		elif self.memaddr == '0' and self.type == 'memory':
			self.memaddr = 'm' + self.content
		elif self.memaddr == '0' and self.type == 'displacement':
			if self.content.find(' ')==-1:
				self.memaddr = 'd' + self.content
			else:
				# if displacement is not an immediate, just save it's a ptr
				self.memaddr = 'd' + 'ptr'
		elif self.memaddr == '0' and self.type == 'phrase':
			# this one creates a lot of trouble later
			#self.memaddr = 'p' + self.content
			# it is better to save somth inocuous like ptr_autoincr, or autoid_phrase
			autoid_phrase += 1
			self.memaddr = 'phrs' + str(autoid_phrase)
		elif self.memaddr == '0':
			autoid+=1
			self.memaddr = '_' + str(autoid)
			#Message(" memaddrr %s and autoid %d\n" % (self.memaddr,autoid))
			
		# to avoid overwriting, first verify not in dict
		if self.memaddr not in results_dict['nodes'].keys():
			results_dict['nodes'][self.memaddr] = {
				'memaddr' : self.memaddr,
				'type' : self.type,
				'content': self.content,
				'mnemonic_content': self.mnemonic_content,
				'raw_content': self.raw_content
			}
		else:
			# if already there, then study how to modify it
			# don't modify type, memaddr 
			# maybe can append data to content...
			if results_dict['nodes'][self.memaddr]['type'] != self.type and \
			    self.type == 'instr':
				"""
					If node was saved as not instr (like data), then when it is saved as a instr it will prevail over other non instr. definitions
				"""
				results_dict['nodes'][self.memaddr]['type']=  self.type	
				results_dict['nodes'][self.memaddr]['content']= self.content
				results_dict['nodes'][self.memaddr]['mnemonic_content']= self.mnemonic_content
				results_dict['nodes'][self.memaddr]['raw_content']= self.raw_content
			elif  results_dict['nodes'][self.memaddr]['type'] != self.type and \
			    results_dict['nodes'][self.memaddr]['type'] == 'instr':
				""" Any previous definition of the node that is instr and the new one is not instr, the previous definition will take precedence
				"""
				pass
			elif  results_dict['nodes'][self.memaddr]['type'] != self.type and \
			    self.type == 'func':
				"""
					if old type is not func(and per if order is not instr either) but new type is func, 
					then overwrite with new_type
				"""
				results_dict['nodes'][self.memaddr]['type']=  self.type	
				results_dict['nodes'][self.memaddr]['content']= self.content
				results_dict['nodes'][self.memaddr]['mnemonic_content']= self.mnemonic_content
				results_dict['nodes'][self.memaddr]['raw_content']= self.raw_content
			elif results_dict['nodes'][self.memaddr]['type'] == 'unknown' and \
			    self.type == 'func':
				"""
					func prevails over unknown
				"""
				results_dict['nodes'][self.memaddr]['type']=  self.type	
				results_dict['nodes'][self.memaddr]['content']= self.content
				results_dict['nodes'][self.memaddr]['mnemonic_content']= self.mnemonic_content
				results_dict['nodes'][self.memaddr]['raw_content']= self.raw_content
			elif  results_dict['nodes'][self.memaddr]['type'] != self.type and \
			    results_dict['nodes'][self.memaddr]['type'] != 'instr':
				"""
					if old type is not instr, 
						and not(old_type=unknown and new_type=func)
					then overwrite with new_type
				"""
				
				results_dict['nodes'][self.memaddr]['type']+= "; "  + self.type
			elif results_dict['nodes'][self.memaddr]['content'] != self.content:
				"""
					Type is the same and content is not the same: then concatenate the content. 
				"""
				Message('overwriting node!\n')
				Message(results_dict['nodes'][self.memaddr]['content'])
				Message("\n")
				Message(self.content)
				Message("\n")
				
				results_dict['nodes'][self.memaddr]['content']+= "; "  + self.content
				# save only the last mnemonic_content
				results_dict['nodes'][self.memaddr]['mnemonic_content']= self.mnemonic_content
				# save only the last raw content
				results_dict['nodes'][self.memaddr]['raw_content']= self.raw_content
				
		
		
		# will be written to disk at the end of the writeGraph function
		
		
		#self.fd_nodes.write('%s {"type": "%s", "content": "%s"}\n' % (self.memaddr, self.type, self.content))
		#Message("save:  %s  %s  %s\n" % (self.memaddr, self.type, self.content))
		
	def saveEdge(self, destination, results_dict):
		
		edgeid = self.memaddr +"-"+ destination.memaddr
		if edgeid not in results_dict['edges'].keys():
			results_dict['edges'][edgeid] = {
				'source': self.memaddr,
				'dest': destination.memaddr,
				'type': '{ "type": "'+destination.type+'" }'
			}
		
		#self.fd_edges.write('%s %s {"type": "%s" }\n' % (self.memaddr, destination.memaddr, destination.type))
		#Message("saveEdge: %s  %s  %s -  %s %s %s\n" % (self.memaddr, self.type, self.content,destination.memaddr, destination.type, destination.content))
	
	def sanitize_content(self):
		"""
			This method sanitized current node's content.
			To be called before saving to disk.
			
			sanitization:
				- remove \\, \, %,
				- remove any '? not for the moment
				- replace " by '
		"""
		#self.content = self.content.replace('"', '\'') # maybe they could be removed also
		self.content = self.content.replace('"', '')
		self.content = self.content.replace('\'', '')
		self.content = self.content.replace('\\n', '')
		self.content = self.content.replace('\\', '')
		self.content = self.content.replace('\%', '')
		

def initializeFolder():
	prefix = "C:\labs\IDA_pro_book\\" 
	prefix = "H:\IDA_pro_book\\"
	prefix = "C:\\big-noisy-dataset\\graphs\\"
	try:
		thedate = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S') 
		folder_name  = AskStr(prefix + "newprogram_"+thedate , "Folder to save files to?")
		# create folder
		os.mkdir(folder_name)
		return folder_name
	except:
		Message(" Error initializing folder!")
		return False

def initializeFiles(folder_name, func):
	try:
		func = get_func(GetFunctionAttr(f, FUNCATTR_START))
		fname = Name(func.startEA)
		
		# remove strange chars
		final_str = ''
		for char in fname:
			if char in string.printable:
				final_str += char
		fname = final_str
		fname = fname.replace('@','')
		fname = fname.replace('?','')
		
		
		# initialize files 
		fd_nodes = open(folder_name + "/"+fname+"_nodes.txt", "w")
		fd_edges = open(folder_name + "/"+fname+"_edges.txt", "w")

		return fd_nodes, fd_edges
		
	except:
		Message(" Error initializing files at %s for %s " % (folder_name,Name(func.startEA)))
		return None, None
		

def instrToStr(instr):
	"""
		Receives a memaddr corresponding to an instruction
		It decodes the instruction and returns its mnemonic
	"""
	instr = DecodeInstruction(instr)
	if instr is not None:
		return instr.get_canon_mnem()
	else:
		return None
			
def processXrefFrom(f, xref, i, current_func_name, fd_nodes, fd_edges):
	global results_dict
	# save the caddr as a node (duplicates can exist)
	#Message(' '.join(dir(xref)))
	# It is misleading, but the xref.to is the address of the origin of this xreffrom
	xref_node = codeNode(str(xref.to), 
						  type="instr", 
						  content=instrToStr(xref.to),
						  fd_nodes = fd_nodes,
						  fd_edges = fd_edges)
		
	#verify = False
	#if xref_node.content is not None and   xref_node.content.find('ret') > -1:
	#	Message(" xref  %s type is %d %s  content: %s \n" % (xref_node.memaddr, xref.type,XrefTypeName(xref.type), xref_node.content) )
	#	verify = True
	
	#Message(" xref type is %d %s \n" % (xref.type,XrefTypeName(xref.type)) )
	if XrefTypeName(xref.type) == 'Ordinary_Flow':
		pass # the xref is instr, and you want to write the edge
	elif xref.type == fl_CF or xref.type == fl_JF :
		# jump or call far
		xref_node.type = "func"
		xref_node.content = Name(xref.to)
		xref_node.mnemonic_content = Name(xref.to)
		xref_node.raw_content =  '' # technically not in the function, so no need for the raw content
		xref_node.saveNode(results_dict)
		
	elif xref.type == fl_CN or xref.type == fl_JN or xref.type == fl_F:
		# jump/call near or ordinary flow
		# compute if it comes from outside the function
		# if it does -> change type to func
		# otherwise do nothing 
	
		# HOW TO COMPUTE: initial func instruction?
		func_start = GetFunctionAttr(f, FUNCATTR_START)
		
		# HOW TO COMPUTE: last func instruction?
		func_end = GetFunctionAttr(f, FUNCATTR_END)
		
		# comparison
		xref_start = GetFunctionAttr(xref.to, FUNCATTR_START)
		inside = xref_start >= func_start and xref_start <= func_end
		#inside = True
		#Message(" xref_start: %s func_start: %s  func_end: %s \n" % (xref_start, func_start, func_end))
		
		# decision making
		if inside:
			# this node wil be deduped in case it is already registered
			xref_node = process_instruction(f,xref.to,results_dict, current_func_name, fd_nodes,fd_edges)
		else:
			#Message(" xref %s computed as xref from outside the function! \n" % xref_node.memaddr)
			xref_node.type = "func"
			xref_node.content =  Name(xref.to)
			xref_node.saveNode(results_dict)
			
	elif XrefTypeName(xref.type).find('Data_')>-1:
		""" memaddress or data that points to this func
			savevd pointer in mem?
		"""
		xref_node = None

		
	return xref_node
			
def get_func_name(opaddr):
	operand_func_start = GetFunctionAttr(opaddr, FUNCATTR_START)
	return Name(operand_func_start)
		
def processOperand(f, op,i, op_position, current_func_name, fd_nodes, fd_edges, verify=False):
	"""
	https://www.hex-rays.com/products/ida/support/idadoc/277.shtml

	 operand is an immediate value  => immediate value
	 operand has a displacement     => displacement
	 operand is a direct memory ref => memory address
	 operand is a register          => register number
	 operand is a register phrase   => phrase number
	 otherwise                      => -1
	
	https://www.hex-rays.com/products/ida/support/idadoc/276.shtml
	
	o_void = 0
 	o_reg = 1
 	o_mem = 2
 	o_phrase = 3
 	o_displ = 4
 	o_imm = 5
 	o_far = 6
 	o_near = 7
 	o_idpspec0 = 8
 	o_idpspec1 = 9
 	o_idpspec2 = 10
 	o_idpspec3 = 11
 	o_idpspec4 = 12
 	o_idpspec5 = 13
	
	"""
	global results_dict
	global operands_dict
	operand_node = codeNode(str(op.addr), 
						  type="data", 
						  content="",
						  fd_nodes = fd_nodes,
						  fd_edges = fd_edges)
						  
	operand_node.mnemonic_content=GetOpnd(i,op_position)
	operand_node.raw_content=GetOpnd(i,op_position)
	#if op.type in [o_displ, o_phrase]:
	if op.type == o_void:
		return None
	elif op.type == o_displ:		
		operand_node.type="displacement"
		operand_node.content=str(op.value)
	elif op.type == o_mem:
		operand_node.type="memory"
		operand_node.content=str(op.value)
		addr_func_name = get_func_name(op.addr)
		if current_func_name == addr_func_name or addr_func_name == '':
			operand_node.mnemonic_content = "memory"
		else:
			operand_node.mnemonic_content = addr_func_name
			Message("\n memory operand, applied addr_func_name and found" + operand_node.mnemonic_content + "\n" )
	elif op.type == o_imm:
		operand_node.type="immediate"
		operand_node.content=str(op.value)
		operand_node.mnemonic_content = "immediate"
	elif op.type == o_reg:
		operand_node.type="register"
		operand_node.content=str(op.reg)
	elif op.type == o_phrase:
		operand_node.type="phrase"
		
		instruc_str = idc.GetDisasm(i)
		phrases = instruc_str.split(',')
		phrase = str(op.phrase)
		if len(phrase)>1 and phrase[0] == ' ':
			phrase = phrase[1:]
		elif len(phrases) > 1:
			phrase = phrases[1]
		operand_node.content=str(phrase) 
		
	elif op.type == o_near or op.type == o_far:
		
		addr_func_name = get_func_name(op.addr)
		if  current_func_name == addr_func_name:
			# it's an instruction -> so make an edge but print mem address
			operand_node = process_instruction(f,op.addr,results_dict, current_func_name, fd_nodes,fd_edges)
		else:
			operand_node.type="func"# should be far
			operand_node.content=addr_func_name
			operand_node.mnemonic_content = addr_func_name
		
	elif op.type == o_idpspec0:
		operand_node.type="idpspec0"
		operand_node.content=str(op.value)
		operand_node.mnemonic_content = operand_node.type 
	elif op.type == o_idpspec1:
		operand_node.type="idpspec1"
		operand_node.content=str(op.value)
		operand_node.mnemonic_content = operand_node.type 
	elif op.type == o_idpspec2:
		operand_node.type="idpspec2"
		operand_node.content=str(op.value)
		operand_node.mnemonic_content = operand_node.type 
	elif op.type == o_idpspec3:
		operand_node.type="idpspec3"
		operand_node.content=str(op.value)
		operand_node.mnemonic_content = operand_node.type 
	elif op.type == o_idpspec4:
		operand_node.type="idpspec4"
		operand_node.content=str(op.value)
		operand_node.mnemonic_content = operand_node.type 
	elif op.type == o_idpspec5:
		operand_node.type="idpspec5"
		operand_node.content=str(op.value)
		operand_node.mnemonic_content = operand_node.type 
	else:
		if verify:
			Message(" in type unknown \n")
		operand_node.type="unknown"
		operand_node.content=str(op.value)
		operand_node.mnemonic_content = operand_node.type
		
	operand_node.saveNode(results_dict)
	return operand_node
		
def writeFuncGraphToDisk(results_dict, fd_nodes, fd_edges):
	
	for k,v in results_dict['nodes'].items():
		
		memaddr = v['memaddr']
		type = v['type']
		content = v['content']
		mnemonic_content = v['mnemonic_content']
		raw_content = v['raw_content']
		fd_nodes.write('%s {"type": "%s", "content": "%s", "mnemonic_content": "%s", "raw_content":"%s"}\n' % (memaddr, type, content, mnemonic_content, raw_content))
		
	for k,v in results_dict['edges'].items():
		"""
			{
				'source': self.memaddr,
				'dest': destination.memaddr,
				'type': '{ "type": "'+destination.type+'" }'
			}
		"""
		source = v['source']
		destination = v['dest']
		type = v['type']
		fd_edges.write('%s %s %s\n' % (source, destination, type))
	
def process_instruction(f,i,results_dict, current_func_name, fd_nodes,fd_edges):
	# prepare components of each node and edge list entry
	
	# save the origin caddr as a node (the instruction node)
	instr_node = codeNode(str(i), 
						  type="instr", 
						  content=instrToStr(i),
						  fd_nodes = fd_nodes,
						  fd_edges = fd_edges)
						  
	
	
	# save instr. operands as nodes and save edges too 			
	instr = DecodeInstruction(i)
	operands = [instr_node.content]
	mnemonic_operands =  [instr_node.content]
	op_pos = 0
	for op in instr.Operands:
		
		operand_node = processOperand(f, op,i,op_pos, current_func_name, fd_nodes, fd_edges, False)
		op_pos+=1
		
		
		if operand_node is not None:
			if operand_node.raw_content=="sprintf":
				Message(operand_node.type)
				Message("\n")
				Message(operand_node.content)
				Message("\n")
				Message(instr_node.content)
				Message("\n")
				
			if instr_node.content== "call" and operand_node.type=="memory":
				operand_node.type="func"
				operand_node.mnemonic_content=operand_node.raw_content
				operand_node.content=operand_node.raw_content
				operand_node.saveNode(results_dict)
			
		
			instr_node.saveEdge(operand_node, results_dict)
			if operand_node.type=="instr":
				operands.append(operand_node.memaddr)
				mnemonic_operands.append(operand_node.memaddr)
			else:
				operands.append(operand_node.content)
				mnemonic_operands.append(operand_node.mnemonic_content)
		

	instr_node.content = " ".join(operands)
	instr_node.mnemonic_content = " ".join(mnemonic_operands)
	instr_node.raw_content = idc.GetDisasm(i)
	#Message("\n                    instr by ops:  ")
	#Message(instr_node.content)
	#Message("\n                    instr       :  ")
	#Message(instr_node.mnemonic_content)
	#Message("\n                    raw instr   :  ")
	#Message(instr_node.raw_content)
	#Message("\n")
	
	instr_node.saveNode(results_dict)
	
		
	# save xrefs From i
	for xref in XrefsFrom(i,0):
		xref_node = processXrefFrom(f, xref,i, current_func_name, fd_nodes, fd_edges)
		if xref_node is not None:
			instr_node.saveEdge(xref_node, results_dict)
			
	return instr_node
			
			
def writeGraph(f, current_func_name,  fd_nodes, fd_edges):
	"""
	    This functions writes nodes and edges with the following format
	    Networkx format:
		- edge list: 1 2 ('weight':7, 'color': green, ...)
			- direct import into Networkx
		- node features file:  node id { 'feat1_name': feat1_val, ...}
			- manually add those features into the nodes once imported from edge_list
	
		now first using a dict, then writing to disk
	"""
	
	global results_dict
	results_dict = { 'nodes': {}, 'edges': {}}
	
	func = get_func(GetFunctionAttr(f, FUNCATTR_START))
	if not func is None:
		
		fname = Name(func.startEA)
		items = FuncItems(func.startEA)	
		
		for i in items:
			process_instruction(f,i,results_dict, current_func_name, fd_nodes,fd_edges)
					
			
		#write func to disk
		writeFuncGraphToDisk(results_dict, fd_nodes, fd_edges)
		
			
			
folder_name = initializeFolder()
if not folder_name:
	Message("Error creating folder!")
	exit()

	
"""

https://www.assemblylanguagetuts.com/x86-assembly-registers-explained/

bit	16 bit	8+8 bit	Segment	Pointer	Index	Status Flags	Control Flags
EAX	AX	AL+AH	CS	SP	SI	CF	SF	TF
EBX	BX	BL+BH	DS	BP	DI	PF	OF	IF
ECX	CX	CL+CH	SS			AF	ZF	DF
EDX	DX	DL+DH	ES					
"""
	
operands_dict = {
	'registers': {
		# ax,cx,dx,bx   8-11
		# rcx,rdx,rbx ?
		0: 'eax',
		1: 'ecx',
		2: 'edx',
		3: 'ebx',
		4: 'esp',
		5: 'ebp',
		6: 'esi',
		7: 'edi',
		8: 'r8',
		9: 'r9',
		10: 'rax',
		11: 'r11d', #'rcx', 
		12: 'r12',
		13: 'r13', 
		14: 'r14', 
		15: 'r15', 
		16: 'al',
		17: 'cl',
		18: 'dl',
		19: 'bl',
		20: 'ah',
		21: 'ch',
		22: 'dh',
		23: 'bh',
		#24: 'spl', 
		25: 'bpl', 
		26: 'sil', 
		27: 'dil', 
		#28: '',
		29: 'es',
		30: 'cs',
		31: 'ss',
		32: 'ds',
		33: 'fs',
		34: 'gs',
		64: 'xmm0',
		65: 'xmm1',
		66: 'xmm2',
		67: 'xmm3',
		68: 'xmm4',
		69: 'xmm5',
		70: 'xmm6',
		71: 'xmm7',
		72: 'xmm8',
		73: 'xmm9',
		74: 'xmm10',
		75: 'xmm11',
		76: 'xmm12',
		77: 'xmm13',
		78: 'xmm14',
		79: 'xmm15',
		
		},
	}
results_dict = {}
funcs = Functions()
for f in funcs:
	func = get_func(GetFunctionAttr(f, FUNCATTR_START))
	if not func is None: # and \
	   #Name(func.startEA)=='Call_Decryption_Routine_45E320': # 'sub_452740':
	   #Name(func.startEA)=='sub_933C00':
	   #Name(func.startEA)=='ERR_peek_last_error_line':
	   #Name(func.startEA)=='sub_404BE0':
	   #Name(func.startEA)=='sub_4049B0':
	   #Name(func.startEA)=='sub_518990': 
	   
		current_func_name = Name(func.startEA)
		fd_nodes, fd_edges = initializeFiles(folder_name,f)
		try:
			autoid = 0
			autoid_phrase = 0
			writeGraph(f, current_func_name, fd_nodes, fd_edges)		
			fd_edges.close()
			fd_nodes.close()
		except  Exception as e:
			fname = Name(func.startEA)
			#Message(' '.join(dir(e)))
			Message(str(e.message))
			Message(" cannot write %s \n" % fname)
	else:
		pass
		#Message(" func is None %s" % f)

for k,v in operands_dict['registers'].items():
	Message(str(k)+' '+str(v)+'\n')
				
# #      Flow types (combine with XREF_USER!):
# fl_CF   = 16              # Call Far
# fl_CN   = 17              # Call Near
# fl_JF   = 18              # jumpto Far
# fl_JN   = 19              # jumpto Near
# fl_F    = 21              # Ordinary flow

# XREF_USER = 32            # All user-specified xref types
                          # # must be combined with this bit
						  					  
# # Data reference types (combine with XREF_USER!):
# dr_O    = ida_xref.dr_O  # Offset
# dr_W    = ida_xref.dr_W  # Write
# dr_R    = ida_xref.dr_R  # Read
# dr_T    = ida_xref.dr_T  # Text (names in manual operands)
# dr_I    = ida_xref.dr_I  # Informational

# add_dref = ida_xref.add_dref
# del_dref = ida_xref.del_dref
# get_first_dref_from = ida_xref.get_first_dref_from
# get_next_dref_from = ida_xref.get_next_dref_from
# get_first_dref_to = ida_xref.get_first_dref_to
# get_next_dref_to = ida_xref.get_next_dref_to