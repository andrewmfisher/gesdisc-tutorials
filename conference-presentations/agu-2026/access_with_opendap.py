# Get IMERG v07 data with Cloud OPeNDAP
# 
# 8/26/2026 JRS

import earthaccess
import xarray as xr

earthaccess.login()

mode = "stream" # stream, download, or opendap

# Search for granules
results = earthaccess.search_data(
    short_name = "GPM_3IMERGDF",
    version = "07" ,
    temporal = ('2000-01-01','2000-01-03')
)
# Note: there is no need for temporal or spatial arguments since each granule has global coverage and we are interested in the entire period of record

# Parse out the OPeNDAP URLs from the "results" object
opendap_urls = []
for item in results: # loop over granules
    for urls in item['umm']['RelatedUrls']: # loop over RelatedURLs
        url = urls['URL'].replace('https','dap4')
        opendap_urls.append(url)

print("the opendap URLS")
# print(opendap_urls)

print("the umm metadata")
# print(results[0]['umm'])
print(len(results))

match mode:
    case "stream":
        # Use earthaccess to stream the data -------------------------------
        
        # print(earthaccess.__version__)
        # print(type(earthaccess.open(results)[0]))
        
        print("streaming")
        files = earthaccess.open(results, provider = "GES_DISC")
        ds = xr.open_mfdataset(files, combine = "by_coords", engine = "h5netcdf")
        
        print(ds)
    case "download":
        # Use earthaccess to download the data -----------------------------
        print("downloading")
        saveloc = "~/Downloads/"
        filelist = earthaccess.download(results, local_path = saveloc)
        print(f"downloaded granules to {saveloc}")
    case "opendap":
        # Use opendap to subset and stream the data ------------------------
        print("opendap")


# Load dataset object and metadata
# print("opening")
# fileobj = earthaccess.open(results)
# print('streaming')
# ds = xr.open_mfdataset(fileobj)

# ds = xr.open_mfdataset(opendap_urls, engine = "pydap")

# ds.head()
