### file transfer:

- open a tunnel on a different terminal:
`ssh -L 1234:172.17.158.31:22  nova.cs.tau.ac.il`

- connect to the host (if needed):
`ssh -p 1234 student1@localhost -o "UserKnownHostsFile=/dev/null" -o "StrictHostKeyChecking=no"`

- copy files:
`scp  -o "UserKnownHostsFile=/dev/null" -o "StrictHostKeyChecking=no" -P 1234 /path/to/local/file student1@localhost:/path/to/desired/remote/location`

### open jupyter notebook
- open a tunnel on a different terminal:
`ssh -L 1234:172.17.158.31:22  nova.cs.tau.ac.il`

- ssh to the machine with
`ssh -p 1234 student1@localhost -o "UserKnownHostsFile=/dev/null" -o "StrictHostKeyChecking=no" -L 8888:localhost:8888`

- run jupyter
`student1@momo:~$ jupyter lab`

- copy the link from the output
`        http://localhost:8889/?token=9f9f1fe167217fd2e8779efb45da5ff73c65c98bb04f31e8&token=9f9f1fe167217fd2e8779efb45da5ff73c65c98bb04f31e8`

- if the jupyter opens on a different port than 8888 you need to redo the proccess with the correct port

- paste the link in the browser

- enjoy
