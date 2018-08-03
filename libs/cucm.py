#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import time
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException, NoAlertPresentException
from selenium.webdriver.remote.remote_connection import LOGGER

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

IMPLICIT_WAIT_TIME = 10

def addLog(f):
    def wrapped_f(*args, **kwargs):
        logger.info("Enter: {}".format(f.__name__))
        result = f(*args, **kwargs)
        logger.info("Exit: {}".format(f.__name__))
        return result
    return wrapped_f

class CucmWebApp(object):
    ccmadmin = "ccmadmin"
    reporting = "cucreports"
    recovery = "drf"
    serviceability = "ccmservice"
    osadmin = "cmplatform"


class Cucm(object):
    def __init__(self, host, cm_username="", cm_password="", os_username="", os_password="", version="11"):
        self.host = host
        self.cm_username = cm_username
        self.cm_password = cm_password
        self.os_username = os_username
        self.os_password = os_password
        self.version = version

        firefox_options = webdriver.FirefoxOptions()
        firefox_options.add_argument('-headless')
        self.browser = webdriver.Firefox(firefox_options=firefox_options)
        self.browser.implicitly_wait(IMPLICIT_WAIT_TIME)

        logger.debug("Initialized browser for CUCM " + self.host)
        self.open_app()

    def open_app(self, app=CucmWebApp.ccmadmin):
        self.browser.get("https://" + self.host + "/" + app + "/")
        time.sleep(1)
        try:
            if app in [CucmWebApp.ccmadmin, CucmWebApp.serviceability, CucmWebApp.reporting]:
                username = self.cm_username
                password = self.cm_password
            else:
                username = self.os_username
                password = self.os_password
            self.browser.find_element_by_name("j_username").send_keys(username)
            self.browser.find_element_by_name("j_password").send_keys(password)
            time.sleep(1)
            self.browser.find_element_by_name("logonForm").submit()
            time.sleep(1)
        except NoSuchElementException:
            logger.info("Possibly already login to " + app)

        try:
            self.browser.find_element_by_link_text("Logout")
            logger.info("Successfully opened " + app)
            return True
        except NoSuchElementException:
            logger.info("Failed to open " + app)
            return False

    def current_app(self):
        try:
            app_nav = self.browser.find_element_by_id("appNav")
            return Select(app_nav).first_selected_option.get_attribute("value")
        except NoSuchElementException:
            logger.error("Possibly not on CUCM web page.")
            return None

    def find_phone(self, phone_mac):
        if self.current_app() != CucmWebApp.ccmadmin:
            if not self.open_app(CucmWebApp.ccmadmin):
                logger.error("Unable to open CUCM CM Admin web console")
                return None

        self.browser.get("http://" + self.host + "/ccmadmin/phoneFindList.do")
        Select(self.browser.find_element_by_id("searchField0")).select_by_visible_text("Device Name")
        search_box = self.browser.find_element_by_id("searchString0")
        search_box.clear()
        search_box.send_keys("SEP" + phone_mac.upper())
        self.browser.find_element_by_name("findButton").click()

        device = self.browser.find_element_by_xpath(
            "//table[@summary='Find List Table Result']/tbody/tr[2]")
        phone = dict(name=device.find_element_by_xpath(".//td[3]/a").text,
                     link=device.find_element_by_xpath(".//td[3]/a").get_attribute("href"),
                     description=device.find_element_by_xpath(".//td[4]").text,
                     status=device.find_element_by_xpath(".//td[7]").text,
                     IP=device.find_element_by_xpath(".//td[8]").text)
        return phone

    def operate_in_pop_up_window(self, callback):
        now_handle = self.browser.current_window_handle
        all_handles = self.browser.window_handles
        for handle in all_handles:
            if handle != now_handle:
                self.browser.switch_to.window(handle)
                time.sleep(1)
                callback()
                break
        self.browser.switch_to.window(now_handle)

    def save_and_apply(self):
        self.browser.find_element_by_id("2tbllink").click()
        alert = self.browser.switch_to.alert
        alert.accept()
        self.browser.find_element_by_id("6tbllink").click()

        def callback():
            self.browser.find_element_by_name("OK").click()

        self.operate_in_pop_up_window(callback)

    def clear_text_field(self, id):
        self.browser.find_element_by_id(id).clear()
        try:
            alert = self.browser.switch_to.alert
            alert.accept()
        except NoAlertPresentException:
            pass

    @addLog
    def delete_line_dn(self, phone, dn):
        self.browser.get(phone['link'])

        self.browser.find_element_by_partial_link_text(dn).click()
        self.browser.find_element_by_id("2tblimage").click()
        alert = self.browser.switch_to.alert
        alert.accept()
        self.browser.find_element_by_css_selector("#searchDiv1 > input[type=\"button\"]").click()

    @addLog
    def add_line_dn(self, phone, new_dn):
        self.browser.get(phone['link'])
        self.browser.find_element_by_partial_link_text("- Add a new DN").click()
        self.clear_text_field("DNORPATTERN")
        self.browser.find_element_by_id("DNORPATTERN").send_keys(new_dn)
        self.browser.find_element_by_id("1tblimage").click()
        self.browser.find_element_by_id("1tblimage").click()
        self.browser.find_element_by_css_selector("#searchDiv1 > input[type=\"button\"]").click()

    @addLog
    def change_line_dn(self, phone, dn, new_dn):
        self.delete_line_dn(phone, dn)
        self.add_line_dn(phone, new_dn)
        self.save_and_apply()

    @addLog
    def change_line_label(self, phone, dn, label):
        self.browser.get(phone['link'])

        self.browser.find_element_by_partial_link_text(dn).click()
        self.clear_text_field("DISPLAY")
        self.clear_text_field("DISPLAYASCII")
        self.browser.find_element_by_id("DISPLAY").send_keys(label)
        self.clear_text_field("LABEL")
        self.browser.find_element_by_id("LABEL").send_keys(label)
        self.browser.find_element_by_id("1tbllink").click()

    def quit(self):
        logger.debug("Close browser!")
        self.browser.quit()

if __name__ == "__main__":
    #my_cucm = Cucm(host="10.74.63.21", cm_username="1", cm_password="1")
    #my_phone = my_cucm.find_phone("5006AB802B51")
    #my_cucm.change_line_dn(my_phone, "10710", "10711")
    #my_cucm.quit()

    #my_cucm = Cucm(host="10.74.63.21", cm_username="1", cm_password="1")
    #my_phone = my_cucm.find_phone("5006AB802B51")
    #my_cucm.change_line_label(my_phone, "10711", "Xiaobin Zhang")
    #my_cucm.quit()
