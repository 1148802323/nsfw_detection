# 引入模块
from obs import ObsClient

ak = ''
sk = ''

server = ""

# 桶名：

# 创建obsClient实例
download = ObsClient(access_key_id=ak, secret_access_key=sk, server=server)

bucketName = ''
objectKey = '1735737927_306244774686326755_0.png'
downloadPath = './picture.jpg'
# 使用访问OBS
resp = download.getObject(bucketName=bucketName, objectKey=objectKey, downloadPath=downloadPath)
if resp.status < 300:
    print('Get Object Succeeded')
    print('requestId:', resp.requestId)
    print('url:', resp.body.url)
else:
    print('Get Object Failed')
    print('requestId:', resp.requestId)
    print('errorCode:', resp.errorCode)
    print('errorMessage:', resp.errorMessage)
    print(resp)

# 关闭obsClient
download.close()

