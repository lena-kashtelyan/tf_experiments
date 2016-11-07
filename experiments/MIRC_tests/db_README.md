#1. Prepare postgresql databse
	a. sudo apt-get install postgresql libpq-dev postgresql-client postgresql-client-common #Install posetgresql with online installer
	b. sudo -i -u postgres #goes into postgres default user
	c. psql postgres #enter the postgres interface
	d. create role cvpr WITH LOGIN superuser password 'XXX'; #create the admin for this webapp
	e. alter role cvpr superuser; #ensure we are sudo
	f. create database cvpr_mirc_experiment with owner cvpr; #create the webapp's database
	g. \q #quit
	h. psql cvpr -h 127.0.0.1 -d cnn_results #log into the database with the admin credentials
	i. create table cnn_results (_id bigserial primary key, model_name varchar, experiment varchar, attention varchar, category varchar, t1 float, t5 float);
