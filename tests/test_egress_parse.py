from egress_check import parse

SAMPLE = '''\
4711  connect(3, {sa_family=AF_INET, sin_port=htons(8080), sin_addr=inet_addr("127.0.0.1")}, 16) = 0
4711  connect(4, {sa_family=AF_UNIX, sun_path="/run/systemd/resolve/io.systemd.Resolve"}, 42) = 0
4712  connect(5, {sa_family=AF_INET6, sin6_port=htons(443), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "2a01:4f8::1", &sin6_addr), sin6_scope_id=0}, 28) = -1 EINPROGRESS
4712  connect(6, {sa_family=AF_INET, sin_port=htons(53), sin_addr=inet_addr("192.168.1.1")}, 16) = 0
'''


def test_parse_strace():
    inet, unix = parse(SAMPLE)
    assert inet == [("127.0.0.1", 8080), ("2a01:4f8::1", 443), ("192.168.1.1", 53)]
    assert unix == {"/run/systemd/resolve/io.systemd.Resolve"}
