#coding=utf8
import sys
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import datetime
import ssl
import time

ssl._create_default_https_context = ssl._create_unverified_context

# 全局变量
vulnCount = 0
codeLinkCount = 0
data_log = open("./data/CVE-Scraper_all_2023.dat", "w+", encoding='utf-8')
error_log = open("./Log/main_log_all.log", "a+")


# 在发生异常时进行重试
def retry_request(url, headers, max_retries=5):
    for i in range(max_retries):
        try:
            response = urlopen(Request(url, headers=headers))
            return response.read()
        except Exception as e:
            print(f"Error in retry_request: {e}")
            print(f"Retrying request... (Attempt {i+1}/{max_retries})")
            time.sleep(5)  # 等待一段时间后重试
    print("Max retries exceeded. Unable to complete request.")
    return None
def log_data(CVEID, cvssscore, basescore, baseseverity):
    global vulnCount
    print("Logging cell data...")
    vulnCount += 1
    print("VULNERABILITIES FOUND: " + str(vulnCount))
    data_log.write('{\n\t"CVE ID":"' + CVEID + '",\n\t"CVSS Score":"' +
                   cvssscore +
                   '",\n\t"Base Score":"' + basescore +
                   '",\n\t"Base Severity":"' + baseseverity +
                   '"\n}\n\n')
    data_log.flush()


def find_code_link(CVEPage):
    global codeLinkCount
    try:
        cveSoup = BeautifulSoup(urlopen(Request(CVEPage, headers={'User-Agent': 'Mozilla/5.0'})).read(), 'html.parser')
        linkStr = ""
        referTable = cveSoup.find('div', style='overflow-x: scroll').find('ul', class_='list-group rounded-0')
        row = referTable.findAll('li', class_="list-group-item border-0 border-top list-group-item-action")
        for cell in row:
            link = cell.find('a')['href']
            if "github.com" in link and "commit" in link:
                codeLinkCount += 1
                linkStr += cell.find('a')['href']
                print("codeLinkCount:" + str(codeLinkCount))
        return linkStr
    except Exception as e:
        print("Error in find_code_link:", e)
        return ""


def log_message(msg):
    timestamp = str(datetime.datetime.now())
    error_log.write(timestamp + ":\t" + msg + "\n")


def record_cve_data(pageURL):
    log_message("scrape extracting from: " + pageURL + "\n")
    try:
        pageSoup = BeautifulSoup(urlopen(Request(pageURL, headers={'User-Agent': 'Mozilla/5.0'})).read(),
                                  'html.parser')
        pageTable = pageSoup.find('div', {'id': 'contentdiv'})
        try:
            CVEID = pageTable.find('h1').find('a').text.strip()
        except:
            CVEID = "NULL"
        try:
            cvsstable = pageTable.find('table', class_='table table-borderless')
            data = "NULL"
            basescore_in = "NULL"
            baseseverity_in = "NULL"
            for row in cvsstable.find_all('tr'):
                cells = row.find_all('td', class_='ps-2')
                for i, cell in enumerate(cells):
                    if i == 0:
                        basescore_in = cell.find('div').get_text().strip()
                    elif i == 1:
                        baseseverity_in = cell.get_text().strip()
                    links = cell.find_all('a')
                    for link in links:
                        data = link.text.strip()
                        if data.startswith("CVSS:3"):
                            break
            cvssscore = data
            basescore = basescore_in
            baseseverity = baseseverity_in
        except:
            cvssscore = "NULL"
            basescore = "NULL"
            baseseverity = "NULL"

        print("\n\n")
        print("===")
        print("CVE ID:\t\t\t\t" + CVEID)
        print("CVSS Score:\t\t\t" + cvssscore)
        print("Base Score:\t\t\t" + basescore)
        print("Base Severity:\t\t\t" + baseseverity)

        log_data(CVEID, cvssscore, basescore, baseseverity)
    except Exception as e:
        print("Error in record_cve_data:", e)
        log_message("Error extracting from: " + pageURL + "\n")


# 修改 scrape_cve_data 函数来使用重试机制
def scrape_cve_data():
    pageURL = "https://www.cvedetails.com/browse-by-date.php"
    log_message("Scrape starting up... root page: " + pageURL)
    yearlyReports = []
    try:
        catalogSoup = BeautifulSoup(retry_request(pageURL, {'User-Agent': 'Mozilla/5.0'}), 'html.parser')
        catalogDiv = catalogSoup.select_one('#contentdiv > div > main > div.bg-white.ps-2')
        if catalogDiv:
            for ul in catalogDiv.find_all('ul'):
                first_li = ul.find('li')
                if first_li:
                    a_tag = first_li.find('a')
                    if a_tag:
                        year_value = a_tag.text.strip()
                        if year_value.isdigit() and 2023 <= int(year_value) <= 2023:
                            year_href = a_tag['href']
                            print("Found year at: https://www.cvedetails.com" + year_href + "\n")
                            yearlyReports.append(
                                {"year": year_value, "url": "https://www.cvedetails.com" + year_href})
    except Exception as e:
        print("Error in scrape_cve_data:", e)
        log_message("Error scraping: " + pageURL)

    print("\n === Years discovered. Grabbing pages for each year ===\n\n")
    try:
        for yearURL in yearlyReports:
            print("Scraping year: " + yearURL['year'] + "\n")
            yearTableSoup = BeautifulSoup(retry_request(yearURL['url'], {'User-Agent': 'Mozilla/5.0'}), 'html.parser')
            pageIndex = yearTableSoup.find('div', {'id': 'pagingb'}, class_='paging')
            for page in pageIndex.find_all('a', href=True):
                pageURL = ("https://www.cvedetails.com" + page['href'])
                print("Scraping page: " + pageURL + "\n")
                pageSoup = BeautifulSoup(retry_request(pageURL, {'User-Agent': 'Mozilla/5.0'}), 'html.parser')
                pageTable = pageSoup.find('div', {'id': 'searchresults'})
                for row in pageTable.find_all('h3', class_="col-md-4 text-nowrap"):
                    cveid = row.find('a').text.strip()
                    print("extracting from: " + "https://www.cvedetails.com/cve/" + str(cveid))
                    record_cve_data("https://www.cvedetails.com/cve/" + str(cveid))
    except Exception as e:
        print("Error in scrape_cve_data:", e)


def main(argv):
    print("\n==== CVE-Scraper ====")
    print("==== Main.py ====\n")
    print("PYTHON VERSION:\t\t" + sys.version)
    log_message("CVE-Scraper Starting up...")

    scrape_cve_data()
    log_message("Scrape complete")


if __name__ == '__main__':
    main(sys.argv[1:])
