

CREATE DATABASE uel;
SHOW DATABASES;
USE unswdatabase;



/* CREATING THE UNSW-NB15 BIG DATA TABLE */

CREATE TABLE IF NOT EXIST unsw(
	-- specifiy table columns and data types
	srcip STRING, sport int, dstip STRING, dsport int, proto STRING, state STRING, dur FLOAT,
	sbytes INT, dbytes INT, sttl INT, dttl INT, sloss INT, dloss INT, service string, Sload FLOAT,
	Dload FLOAT, Spkts INT, Dpkts INT, swin INT, dwin INT, stcpb BIGINT, dtcpb BIGINT, smeansz INT,
	dmeansz INT, trans_depth INT, res_bdy_len INT, Sjit FLOAT, Djit FLOAT, Stime BIGINT, Ltime BIGINT, 
	Sintpkt FLOAT, Dintpkt FLOAT, tcprtt FLOAT, synack FLOAT, ackdat FLOAT, is_sm_ips_ports INT,
	ct_state_ttl INT, ct_flw_http_mthd INT, is_ftp_login INT, ct_ftp_cmd INT, ct_srv_src INT,
	ct_srv_dst INT, ct_dst_ltm INT, ct_src_ltm INT, ct_src_dport_ltm INT, ct_dst_sport_ltm INT,
	ct_dst_src_ltm INT, attack_cat string, Label INT)
	COMMENT "UNSW-NB15 Master Dataset" -- give some description of the data stored in the table
	ROW FORMAT DELIMITED FIELDS TERMINATED BY "," -- specify the delimiter
	LINES TERMINATED BY "\n" -- specify line breaks
	STORED AS TEXTFILE -- specify storage format
	LOCATION "/tmp/dataset/"; -- specify location of data that will be loaded into the table




CREATE TABLE clean_table as SELECT attack_cat FROM unsw;



SELECT attack_cat,
count(*) as count from clean_table
WHERE attack_cat != "None"
GROUP BY attack_cat
ORDER BY 2
DESC


CREATE TABLE service_unsw 
as SELECT IF(service == "-", regexp_replace(service, "-", "unused"), service) from unsw;


SELECT service, 
count(*) as frequency from service_unsw WHERE service != "unused" 
GROUP BY service ORDER BY 2 DESC LIMIT 5

