combinations2 = [


    ('gui','config'),
    ('gui',),

    ('network','send'),
    ('network','parse'),
    ('network','config'),
    ('network',),

    ('disk','file'),
    ('disk','read'),
    ('disk','write'),
    ('disk',),


    ('cryptography','encrypt'),
    ('cryptography','config'),
    ('cryptography',),

    ('datastruct',),

    ('memory','config'),
    ('memory','read'),
    ('memory','write'),
    ('memory',),


    ('process','update'),
    ('process','sync'),
    ('process','config'),
    ('process',),

    ('users',),

    ('computation',),


    ]


combinations = [


    ('gui','hide'),
    ('gui','show'),
    ('gui','config'),
    ('gui','work'),
    ('gui','update'),
    ('gui',),

    ('network','send'),
    ('network','receive'),
    ('network','config'),
    ('network','save'),
    ('network','data'),
    ('network','file'),
    ('network','verify'),
    ('network','answer'),
    ('network','get'),
    ('network',),


    ('disk','work'),
    ('disk','read'),
    ('disk','write'),
    ('disk','delete'),
    ('disk','verify'),
    ('disk','parse'),
    ('disk','match'),
    ('disk',),

    ('cryptography','work'),
    ('cryptography','compute'),
    ('cryptography','send'),
    ('cryptography','encrypt'),
    ('cryptography','verify'),
    ('cryptography','config'),
    ('cryptography',),

    ('datastruct','work'),
    ('datastruct','config'),
    ('datastruct',),

    ('memory','config'),
    ('memory','work'),
    ('memory','read'),
    ('memory','write'),
    ('memory','update'),
    ('memory','delete'),
    ('memory','create'),
    ('memory','verify'),
    ('memory',),



    ('process','start'),
    ('process','stop'),
    ('process','create'),
    ('process','update'),
    ('process','delete'),
    ('process','sync'),
    ('process',),

    ('users',),
    ('users','create'),
    ('users','update'),
    ('users','delete'),
    ('users','verify'),
    ('users','config'),
    ('users','save'),

    ('computation',),
    ('computation','work'),


    ]

filename_topics = {
    
    'network': [
        'gnutls','openssl','nginx','http',
        ],
    'cryptography': [
        'tls','ssl','ssh','puttygen','nettle'],
    'disk': ['sqlite','regedit','database',],
    'gui':['filezilla',],
    'computation': ['gmp','nettle',]
}

topics = {
    'network': [
        'server',
        'post',
        #'get',
        'http','ftp','smtp','imap','pop3',
        'session',
        'sock','socket','stream','content',
        'cookie','peer','upstream','hook','webhook',
        'vhost','host',
    ],
    'disk': [
        'storage',
        'file','handler',
        'encode',
        'printf',
        'sqlite3'

    ],
    'cryptography': [
        'key',
        'decrypt',
        'tls','ssl','gnutls','ssh',
        'AES','ecdsa','encrypt','cipher','decipher','spec',
        'hash','hmac','md5','sha',
        'dsa','rsa','rng','random','rand',
        'certificate','asn1','des','edhkex','private',
        'issuer','certificate','cert','pkcs','digest',
        'ecdh','context','cbc','x509','pubkey','pub',
        'ctime','des3','twofish256','pem',
        'digest','puttygen','ocsp',


    ],
    'datastruct': [
        'hashtable',
        'table',
        'hash',
        'data',
        'tree',
        'node',
        'queue',
        'varbuf',
        'vector','recursive',
        'iterat'

    ],
    'memory': [
        'page',
        'cache',
        'stack',
        'heap',
        'swap',
        'memcat',
        'strmemcat',
        'alloc','malloc','free',
        



    ],
    'gui': [ 
        'window',
        'event',
        'handler',
        'bitmap',
        'prompt',
        'height',
        'size',
        #'capture',
        'frame',
        'View', 
        'scroll',
        'filezilla',
        'dialog',

    ],
    'process': [
        'time',
        'thread',
        'process',
        'program',
        'mutex','lock','throw',
        'error',
        'core','dump',
        'ctors','dtors',
        'log','status','state',
        'conf'


    ],
    'users': [
        'auth', 'user','authentication','account','identity','password','pass','group'
        ,'name','id',
    ],
    'computation': [
        'compute','math','calculus','derive','summation','sign'
    ]

}

tasks = {
    'config': [
        'conf','config','configuration','param', 'parameter',
        'helper','info','startup','reg','registry','policy'
    ],
    'verify': [
        'check',
        'checklist',
    ],
    'work': [
        'work','do','job','task'
    ],
    'save': [
        'save',
    ],  
    'delete': [
        'del','delete','remove',
    ],
    'update': [
        'set','insert'
    ],
    'create': [
        'create',
    ],
    'get': [
        'get','request','req','capture','load',
    ],
    'answer': [
        'answer','response',
        ],
    'read': [
        'read',
    ],
    'write': [
        'write','wrt','set',
    ],  
    'send': [
        'send','snd'
    ],
    'receive': [
        'receive','rcv','receiv'
    ],
    'compute': [
        'mul','div','sum','sub','add','xor'
    ],
    'show': [
        'show'
    ],
    'hide': [
        'hide'
    ],
    'start': [
        'start'
    ],
    'stop': [
        'stop','finish'
    ],
    'sync': [
        'sync',
        'wait'
    ],
    'parse': [
        'parse',
    ],
    'match': [
        'match'
    ],
    'encrypt': [
        'encrypt','cipher','decrypt','decipher','decryption', 'encryption'
        ],
    'data': ['data','database','db','dat'],
    'file': ['file','register','registry','binary']
}