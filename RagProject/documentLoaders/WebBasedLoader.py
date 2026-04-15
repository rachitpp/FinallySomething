from langchain_community.document_loaders import WebBaseLoader

url =  "https://www.apple.com/in/macbook-pro/?afid=p240%7Cgo~cmp-11182149775~adg-109263621973~ad-784581523341_kwd-987393509~dev-c~ext-~prd-~mca-~nt-search&cid=aos-in-kwgo-txt-mac-mac--"

WebData = WebBaseLoader(url)

WebDocuments = WebData.load()


# Printing the length of the web documents
print(len(WebDocuments))
# The length of the Web Documents depends on the number of website that we have sent to the loader
# 1 website = 1 content and metadata pair