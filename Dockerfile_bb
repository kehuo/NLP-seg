FROM docker-reg.basebit.me:5000/base/centos7_py3_dev:1.0.0 as builder
RUN cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
ENV LANG en_US.UTF-8

RUN mkdir -p /root/.ssh
ADD ./id_rsa /root/.ssh/id_rsa
ADD ./resource/pip.conf /root/.pip/

RUN chmod 600 /root/.ssh/id_rsa && echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config

WORKDIR /project

ADD ./requirements.txt ./requirements.txt

RUN source ~/.bash_profile && pyenv shell py3_env && pip install -r ./requirements.txt

ADD ./ ./

RUN source ~/.bash_profile && pyenv shell py3_env && pyinstaller nlp.spec

RUN rm -rf /root/.ssh/id_rsa && rm -rf /root/.ssh/known_hosts

FROM docker-reg.basebit.me:5000/base/centos_cuda:8.0_1

RUN yum install -y epel-release &&  yum install -y nginx
ENV LANG en_US.UTF-8
ENV RUN_TYPE api

WORKDIR /project
RUN cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && mkdir -p /project/nlp_seg/dict && mkdir -p /etc/nginx/data/run
COPY --from=builder /project/dist/nlp ./
COPY --from=builder /project/start.sh ./
COPY --from=builder /project/nlp_seg/dict/* ./nlp_seg/dict/
COPY --from=builder /project/resource/nginx.conf /etc/nginx/
COPY --from=builder /project/resource/default.conf /etc/nginx/conf.d/

CMD ["./start.sh"]

