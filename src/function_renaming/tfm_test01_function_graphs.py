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

class codeNode():
	def __init__(self, memaddr, type, content, fd_nodes=None, fd_edges=None):
		self.memaddr = memaddr
		self.type = type 
		self.content = content
		self.fd_nodes = fd_nodes
		self.fd_edges = fd_edges
	
	def saveNode(self):
		self.fd_nodes.write('%s {"type": "%s", "content": "%s"}\n' % (self.memaddr, self.type, self.content))
		
	def saveEdge(self, destination):
		self.fd_edges.write('%s %s {"type": "%s" }\n' % (self.memaddr, destination.memaddr, destination.type))

def initializeFolder():
	prefix = "C:\labs\IDA_pro_book\\" 
	prefix = "H:\IDA_pro_book\\"
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
		
		# initialize files 
		fd_nodes = open(folder_name + "/"+fname+"_nodes.txt", "w")
		fd_edges = open(folder_name + "/"+fname+"_edges.txt", "w")

		return fd_nodes, fd_edges
		
	except:
		Message(" Error initializing files at %s" % folder_name)
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
			
def processXrefFrom(f, xref, i, fd_nodes, fd_edges):
	# save the caddr as a node (duplicates can exist)
	xref_node = codeNode(str(xref.to), 
						  type="instr", 
						  content=instrToStr(xref.to),
						  fd_nodes = fd_nodes,
						  fd_edges = fd_edges)
	
	if xref.type == fl_CF or xref.type == fl_JF :
		# jump or call far
		xref_node.type = "func"
		xref_node.content = Name(xref.to)
	elif xref.type == fl_CN or xref.type == fl_JN or xref.type == fl_F:
		# jump/call near or ordinary flow
		# compute if it comes from outside the function
		# if it does -> change type to func
		# otherwise do nothing 
	
		# HOW TO COMPUTE: initial func instruction?
		func_start = get_func(GetFunctionAttr(f, FUNCATTR_START))
		
		# HOW TO COMPUTE: last func instruction?
		func_end = get_func(GetFunctionAttr(f, FUNCATTR_END))
		
		# comparison
		inside = xref.to >= func_start and xref.to <= func_end
		
		# decision making
		if inside:
			pass # do nothing
		else:
			xref_node.type = "func"
			xref_node.content = Name(xref.to)

	xref_node.saveNode()
	return xref_node
			
def processOperand(f, op,i, fd_nodes, fd_edges):
	"""
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

	operand_node = codeNode(str(op.addr), 
						  type="data", 
						  content="",
						  fd_nodes = fd_nodes,
						  fd_edges = fd_edges)
						  
	#if op.type in [o_displ, o_phrase]:
	if op.type == o_void:
		return None
	elif op.type == o_displ:		
		# mem ref[BaseReg + index Reg + displ] 
		# mem ref[BaseReg + index Reg] 
		
		#f7maddr.write("%d\n" % (effectivemaddr))
		#f6datamaddr.write("%d,%s,%d\n" % (ea,theinstruction,effectivemaddr))
		
		# alternatively save as an immediate displ or base reg?
		#pass
		operand_node.type="displacement"
		operand_node.content=str(op.value)
	elif op.type == o_mem:
		operand_node.type="memory"
		operand_node.content=str(op.value)
	elif op.type == o_imm:
		operand_node.type="immediate"
		operand_node.content=str(op.value)
	elif op.type == o_reg:
		operand_node.type="register"
		operand_node.content=str(op.reg)
	elif op.type == o_phrase:
		operand_node.type="phrase"
		operand_node.content=str(op.phrase)
	else:
		operand_node.type="unkown"
		operand_node.content=str(op.value)
	
	operand_node.saveNode()
	return operand_node
		
	
def writeGraph(f,  fd_nodes, fd_edges):
	"""
	    This functions writes nodes and edges with the following format
	    Networkx format:
		- edge list: 1 2 ('weight':7, 'color': green, ...)
			- direct import into Networkx
		- node features file:  node id { 'feat1_name': feat1_val, ...}
			- manually add those features into the nodes once imported from edge_list
	
	"""
	func = get_func(GetFunctionAttr(f, FUNCATTR_START))
	if not func is None:
		fname = Name(func.startEA)
		items = FuncItems(func.startEA)	
		
		for i in items:
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
			for op in instr.Operands:
				operand_node = processOperand(f, op,i, fd_nodes, fd_edges)
				if operand_node is not None:
					instr_node.saveEdge(operand_node)
					operands.append(operand_node.content)
			
			instr_node.content = " ".join(operands)
			instr_node.saveNode()
				
			# save xrefs From i
			for xref in XrefsFrom(i,0):
				xref_node = processXrefFrom(f, xref,i, fd_nodes, fd_edges)
				instr_node.saveEdge(xref_node)
		
			
			
folder_name = initializeFolder()
if not folder_name:
	Message("Error creating folder!")
	exit()

funcs = Functions()
for f in funcs:
	fd_nodes, fd_edges = initializeFiles(folder_name,f)
	writeGraph(f, fd_nodes, fd_edges)		
	fd_edges.close()
	fd_nodes.close()
			

			

				
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