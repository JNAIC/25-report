#include "Library.h"
#include <iomanip>
#include <sstream>
#include <limits>
#include <algorithm>
#include <cstdlib>
#include <ctime>
using namespace std;  // 仅在 .cpp 中使用

// ==================== 日期工具函数 ====================
bool operator<(const Date& a, const Date& b) {
    if (a.year != b.year) return a.year < b.year;
    if (a.month != b.month) return a.month < b.month;
    return a.day < b.day;
}

// 将日期转换为"天数"(简化：每月30天，每年360天)
static int toDays(const Date& d) {
    return d.year * 360 + d.month * 30 + d.day;
}

Date addDays(Date d, int days) {
    int total = toDays(d) + days;
    d.year  = total / 360;
    total  %= 360;
    d.month = total / 30;
    total  %= 30;
    d.day   = total;
    if (d.day <= 0) { d.day = 1; }
    if (d.month <= 0) { d.month = 1; }
    if (d.month > 12) { d.month = 12; }
    return d;
}

int daysBetween(const Date& a, const Date& b) {
    return toDays(b) - toDays(a);
}

// ==================== Media 基类 ====================
Media::Media(string id, string title, int year)
    : id(id), title(title), year(year), status(MediaStatus::AVAILABLE) {}

Media::~Media() {}

void Media::display() const {
    string s;
    switch (status) {
        case MediaStatus::AVAILABLE: s = "可借";   break;
        case MediaStatus::BORROWED:  s = "已借出"; break;
        case MediaStatus::RESERVED:  s = "已预约"; break;
        case MediaStatus::LOST:      s = "已遗失"; break;
    }
    cout << "  [" << getTypeName() << "] " << title
         << " (ID:" << id << ")  " << year << "年  状态:" << s;
}

// ==================== Book ====================
Book::Book(string id, string title, int year,
           string author, string publisher, string isbn, int pages)
    : Media(id, title, year), author(author),
      publisher(publisher), isbn(isbn), pages(pages) {}

void Book::display() const {
    Media::display();
    cout << endl
         << "         作者:" << author << "  出版社:" << publisher
         << "  ISBN:" << isbn << "  " << pages << "页"
         << "  可借" << getLoanDays() << "天" << endl;
}

// ==================== Magazine ====================
Magazine::Magazine(string id, string title, int year,
                   int issue, string publisher)
    : Media(id, title, year), issueNumber(issue), publisher(publisher) {}

void Magazine::display() const {
    Media::display();
    cout << endl
         << "         出版社:" << publisher << "  第" << issueNumber << "期"
         << "  可借" << getLoanDays() << "天" << endl;
}

// ==================== DVD ====================
DVD::DVD(string id, string title, int year,
         string director, int duration)
    : Media(id, title, year), director(director), duration(duration) {}

void DVD::display() const {
    Media::display();
    cout << endl
         << "         导演:" << director << "  时长:" << duration << "分钟"
         << "  可借" << getLoanDays() << "天" << endl;
}

// ==================== User ====================
User::User(string uid, string n, string pwd)
    : userId(uid), name(n), password(pwd) {}

User::~User() {}

Librarian::Librarian(string uid, string n, string pwd)
    : User(uid, n, pwd) {}

Reader::Reader(string uid, string n, string pwd)
    : User(uid, n, pwd), borrowedCount(0) {}

// ==================== BorrowRecord ====================
BorrowRecord::BorrowRecord(string mid, string uid, Date bDate, Date dDate)
    : mediaId(mid), userId(uid), borrowDate(bDate),
      dueDate(dDate), returned(false) {}

void BorrowRecord::display(const Media* media, const User* user) const {
    cout << "  馆藏ID:" << mediaId;
    if (media) cout << "  《" << media->getTitle() << "》";
    cout << "  用户:" << (user ? user->getName() : userId);
    cout << "  借阅:" << borrowDate.year << "/"
         << borrowDate.month << "/" << borrowDate.day;
    cout << "  应还:" << dueDate.year << "/"
         << dueDate.month << "/" << dueDate.day;
    if (returned) {
        cout << "  [已归还]";
    } else {
        cout << "  [未还]";
    }
    cout << endl;
}

bool BorrowRecord::isOverdue(Date today) const {
    return !returned && toDays(dueDate) < toDays(today);
}

int BorrowRecord::getOverdueDays(Date today) const {
    if (!isOverdue(today)) return 0;
    return toDays(today) - toDays(dueDate);
}

// ==================== Library 构造函数 ====================
Library::Library()
    : currentUser(nullptr), running(false) {
    today.year = 2026; today.month = 7; today.day = 1;
    initData();
}

// ==================== 主循环 ====================
void Library::run() {
    running = true;
    login();
    if (currentUser == nullptr) {
        cout << "登录失败，退出。" << endl;
        return;
    }

    while (running) {
        mainMenu();
        if (!running) break;

        int choice = readInt(0, 8);
        cls();
        switch (choice) {
            case 0:
                cout << "确定退出? (y/n): ";
                {
                    string yn; cin >> yn;
                    if (yn == "y" || yn == "Y") {
                        running = false;
                        cout << "再见!" << endl;
                    }
                }
                break;
            case 1: listAllMedia(); break;
            case 2: searchMedia();  break;
            case 3: borrowMedia();  break;
            case 4: returnMedia();  break;
            case 5: myBorrows();    break;
            case 6:
                if (currentUser->canManageMedia()) addMedia();
                else cout << "权限不足!" << endl;
                break;
            case 7:
                if (currentUser->canManageUsers()) { addUser(); listUsers(); }
                else cout << "权限不足!" << endl;
                break;
            case 8: advanceDate();  break;
        }
        if (choice != 0) waitEnter();
    }
}

// ==================== 登录 ====================
void Library::login() {
    cls();
    cout << endl;
    cout << "========================================" << endl;
    cout << "      图书馆管理系统 v1.0" << endl;
    cout << "      Library Management System" << endl;
    cout << "========================================" << endl;
    cout << endl;
    cout << "可用账号:" << endl;
    for (const auto& u : users) {
        cout << "  " << u->getId() << " (" << u->getRole()
             << ") 密码:123456" << endl;
    }
    cout << endl;

    cout << "用户ID: ";
    string uid;
    getline(cin, uid);

    cout << "密码: ";
    string pwd;
    getline(cin, pwd);

    for (const auto& u : users) {
        if (u->getId() == uid && u->checkPwd(pwd)) {
            currentUser = u.get();
            cout << endl << "欢迎, " << u->getName()
                 << " (" << u->getRole() << ")!" << endl;
            waitEnter();
            return;
        }
    }
    cout << endl << "用户名或密码错误!" << endl;
    waitEnter();
}

// ==================== 主菜单 ====================
void Library::mainMenu() {
    cls();
    showDate();
    cout << "用户: " << currentUser->getName()
         << " (" << currentUser->getRole() << ")" << endl;
    cout << endl;
    cout << "========== 主菜单 ==========" << endl;
    cout << " 1. 查看所有馆藏" << endl;
    cout << " 2. 搜索馆藏" << endl;
    cout << " 3. 借阅" << endl;
    cout << " 4. 归还" << endl;
    cout << " 5. 我的借阅记录" << endl;

    if (currentUser->canManageMedia()) {
        cout << " 6. 添加馆藏 (管理员)" << endl;
        cout << " 7. 用户管理 (管理员)" << endl;
    } else {
        cout << " --------------------------" << endl;
    }

    cout << " 8. 模拟时间推进" << endl;
    cout << " 0. 退出" << endl;
    cout << "============================" << endl;
    cout << "请选择: ";
}

// ==================== 查看所有馆藏 ====================
void Library::listAllMedia() {
    showDate();
    if (mediaItems.empty()) {
        cout << "馆藏为空。" << endl;
        return;
    }
    cout << "===== 馆藏总览 (共" << mediaItems.size() << "件) =====" << endl;
    for (const auto& m : mediaItems) {
        m->display();   // ★ 多态调用
    }
}

// ==================== 搜索馆藏 ====================
void Library::searchMedia() {
    cout << "===== 搜索馆藏 =====" << endl;
    cout << "输入关键词(标题/作者/ISBN, 留空=全部): ";
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    string kw;
    getline(cin, kw);

    if (kw.empty()) { listAllMedia(); return; }

    // 转小写方便比较
    string lower = kw;
    transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    int found = 0;
    for (const auto& m : mediaItems) {
        string titleLower = m->getTitle();
        transform(titleLower.begin(), titleLower.end(), titleLower.begin(), ::tolower);

        bool match = (titleLower.find(lower) != string::npos);

        // 如果是图书，还搜索作者和ISBN
        Book* bk = dynamic_cast<Book*>(m.get());
        if (bk != nullptr) {
            string authLower = bk->getAuthor();
            transform(authLower.begin(), authLower.end(), authLower.begin(), ::tolower);
            if (authLower.find(lower) != string::npos) match = true;
            if (bk->getIsbn().find(lower) != string::npos) match = true;
        }

        // DVD 搜索导演
        DVD* dvd = dynamic_cast<DVD*>(m.get());
        if (dvd != nullptr) {
            string dirLower = dvd->getDirector();
            transform(dirLower.begin(), dirLower.end(), dirLower.begin(), ::tolower);
            if (dirLower.find(lower) != string::npos) match = true;
        }

        if (match) {
            m->display();
            found++;
        }
    }
    cout << endl << "共找到 " << found << " 条结果。" << endl;
}

// ==================== 添加馆藏 ====================
void Library::addMedia() {
    cout << "===== 添加馆藏 =====" << endl;
    cout << "类型:" << endl;
    cout << " 1. 图书" << endl;
    cout << " 2. 杂志" << endl;
    cout << " 3. DVD" << endl;
    cout << "选择 (0=返回): ";
    int ts = readInt(0, 3);
    if (ts == 0) return;

    cout << endl;
    cin.ignore(numeric_limits<streamsize>::max(), '\n');

    string id, title;
    int year;
    cout << "ID (如 BOOK-001): "; getline(cin, id);
    cout << "题名: ";             getline(cin, title);
    cout << "出版年份: ";         year = readInt(1900, 2100);

    unique_ptr<Media> item;
    switch (ts) {
        case 1: {
            string author, publisher, isbn;
            int pages;
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "作者: ";     getline(cin, author);
            cout << "出版社: ";   getline(cin, publisher);
            cout << "ISBN: ";     getline(cin, isbn);
            cout << "页数: ";     pages = readInt(1, 99999);
            item = make_unique<Book>(id, title, year, author, publisher, isbn, pages);
            break;
        }
        case 2: {
            int issue;
            string publisher;
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "出版社: ";   getline(cin, publisher);
            cout << "期号: ";     issue = readInt(1, 9999);
            item = make_unique<Magazine>(id, title, year, issue, publisher);
            break;
        }
        case 3: {
            string director;
            int duration;
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "导演: ";     getline(cin, director);
            cout << "时长(分钟): "; duration = readInt(1, 999);
            item = make_unique<DVD>(id, title, year, director, duration);
            break;
        }
    }

    mediaItems.push_back(move(item));
    cout << "馆藏已添加!" << endl;
}

// ==================== 删除馆藏 ====================
void Library::removeMedia() {
    cout << "===== 删除馆藏 =====" << endl;
    for (size_t i = 0; i < mediaItems.size(); i++) {
        cout << " " << (i + 1) << ". ";
        mediaItems[i]->display();
    }
    if (mediaItems.empty()) {
        cout << "没有馆藏可删。" << endl;
        return;
    }
    cout << "选择 (0=返回, 1-" << mediaItems.size() << "): ";
    int sel = readInt(0, (int)mediaItems.size());
    if (sel == 0) return;

    // 检查是否被借出
    string mid = mediaItems[sel - 1]->getId();
    for (auto& r : borrowRecords) {
        if (r.getMediaId() == mid && !r.isReturned()) {
            cout << "该馆藏已被借出，无法删除！" << endl;
            return;
        }
    }

    cout << "已删除: " << mediaItems[sel - 1]->getTitle() << endl;
    mediaItems.erase(mediaItems.begin() + (sel - 1));
}

// ==================== 借阅 ====================
void Library::borrowMedia() {
    cout << "===== 借阅 =====" << endl;

    // 检查当前用户是否是读者
    if (!dynamic_cast<Reader*>(currentUser)) {
        cout << "只有读者可以借阅！（请以读者身份登录）" << endl;
        return;
    }
    Reader* reader = dynamic_cast<Reader*>(currentUser);

    // 检查借阅上限
    if (reader->getBorrowedCount() >= reader->getMaxBorrow()) {
        cout << "借阅已达上限(" << reader->getMaxBorrow() << "本)，请先归还！" << endl;
        return;
    }

    // 显示可借馆藏
    int idx = 0;
    vector<Media*> availableList;
    for (auto& m : mediaItems) {
        if (m->getStatus() == MediaStatus::AVAILABLE) {
            availableList.push_back(m.get());
            idx++;
            cout << " " << idx << ". ";
            m->display();
        }
    }
    if (availableList.empty()) {
        cout << "没有可借的馆藏。" << endl;
        return;
    }

    cout << endl << "选择 (0=返回, 1-" << idx << "): ";
    int sel = readInt(0, idx);
    if (sel == 0) return;

    Media* m = availableList[sel - 1];
    m->setStatus(MediaStatus::BORROWED);
    reader->incBorrowed();

    Date due = addDays(today, m->getLoanDays());
    borrowRecords.push_back(
        BorrowRecord(m->getId(), currentUser->getId(), today, due));

    cout << endl << "借阅成功！《" << m->getTitle() << "》" << endl;
    cout << "应还日期: " << due.year << "/" << due.month << "/" << due.day << endl;
}

// ==================== 归还 ====================
void Library::returnMedia() {
    cout << "===== 归还 =====" << endl;

    // 找当前用户未还的记录
    int idx = 0;
    vector<BorrowRecord*> unreturned;
    for (auto& r : borrowRecords) {
        if (r.getUserId() == currentUser->getId() && !r.isReturned()) {
            unreturned.push_back(&r);
            idx++;
            Media* m = findMedia(r.getMediaId());
            cout << " " << idx << ". ";
            r.display(m, currentUser);
        }
    }
    if (unreturned.empty()) {
        cout << "没有未归还的借阅记录。" << endl;
        return;
    }

    cout << endl << "选择要归还的 (0=返回, 1-" << idx << "): ";
    int sel = readInt(0, idx);
    if (sel == 0) return;

    BorrowRecord* rec = unreturned[sel - 1];
    rec->setReturned();

    // 恢复馆藏状态
    Media* m = findMedia(rec->getMediaId());
    if (m) m->setStatus(MediaStatus::AVAILABLE);

    // 更新读者借阅计数
    if (dynamic_cast<Reader*>(currentUser)) {
        dynamic_cast<Reader*>(currentUser)->decBorrowed();
    }

    // 检查是否逾期
    int overdue = rec->getOverdueDays(today);
    cout << endl << "归还成功！" << endl;
    if (overdue > 0) {
        double fine = overdue * 0.5;  // 每天0.5元
        cout << "!!! 逾期 " << overdue << " 天，罚款 " << fine << " 元 !!!" << endl;
    }
}

// ==================== 我的借阅 ====================
void Library::myBorrows() {
    cout << "===== 我的借阅记录 =====" << endl;
    int count = 0;
    for (const auto& r : borrowRecords) {
        if (r.getUserId() == currentUser->getId()) {
            Media* m = findMedia(r.getMediaId());
            r.display(m, currentUser);
            if (!r.isReturned()) {
                int od = r.getOverdueDays(today);
                if (od > 0)
                    cout << "    ⚠ 已逾期 " << od << " 天!" << endl;
            }
            count++;
        }
    }
    if (count == 0) cout << "暂无借阅记录。" << endl;
}

// ==================== 全部借阅（管理员） ====================
void Library::allBorrows() {
    cout << "===== 全部借阅记录 =====" << endl;
    if (borrowRecords.empty()) {
        cout << "暂无借阅记录。" << endl;
        return;
    }
    for (const auto& r : borrowRecords) {
        Media* m = findMedia(r.getMediaId());
        User*  u = findUser(r.getUserId());
        r.display(m, u);
    }
}

// ==================== 用户管理 ====================
void Library::addUser() {
    cout << "===== 添加用户 =====" << endl;
    cout << "类型:" << endl;
    cout << " 1. 图书管理员" << endl;
    cout << " 2. 读者" << endl;
    cout << "选择 (0=返回): ";
    int ts = readInt(0, 2);
    if (ts == 0) return;

    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    string uid, name, pwd;
    cout << "用户ID: ";   getline(cin, uid);
    cout << "姓名: ";     getline(cin, name);
    cout << "密码: ";     getline(cin, pwd);

    unique_ptr<User> u;
    if (ts == 1) u = make_unique<Librarian>(uid, name, pwd);
    else         u = make_unique<Reader>(uid, name, pwd);

    users.push_back(move(u));
    cout << "用户已添加!" << endl;
}

void Library::listUsers() {
    cout << endl << "===== 用户列表 =====" << endl;
    for (const auto& u : users) {
        cout << "  ID:" << u->getId()
             << "  姓名:" << u->getName()
             << "  角色:" << u->getRole();
        if (dynamic_cast<Reader*>(u.get())) {
            auto r = dynamic_cast<Reader*>(u.get());
            cout << "  已借:" << r->getBorrowedCount()
                 << "/" << r->getMaxBorrow();
        }
        cout << endl;
    }
}

// ==================== 时间推进 ====================
void Library::advanceDate() {
    cout << "===== 模拟时间 =====" << endl;
    showDate();
    cout << endl;
    cout << " 1. +1天" << endl;
    cout << " 2. +7天" << endl;
    cout << " 3. +30天" << endl;
    cout << "选择 (0=返回): ";
    int sel = readInt(0, 3);
    if (sel == 0) return;

    int adv = 0;
    if (sel == 1) adv = 1;
    if (sel == 2) adv = 7;
    if (sel == 3) adv = 30;

    today = addDays(today, adv);
    cout << endl << "当前时间: ";
    showDate();
    cout << endl;

    // 检查是否有逾期但未归还的
    int overdueCount = 0;
    for (const auto& r : borrowRecords) {
        if (r.isOverdue(today)) {
            overdueCount++;
            Media* m = findMedia(r.getMediaId());
            User*  u = findUser(r.getUserId());
            cout << "  ⚠ 逾期: " << (u ? u->getName() : "?")
                 << " → 《" << (m ? m->getTitle() : "?") << "》"
                 << " 逾期" << r.getOverdueDays(today) << "天" << endl;
        }
    }
    if (overdueCount == 0) {
        cout << "  当前无逾期记录。" << endl;
    }
}

// ==================== 工具函数 ====================
void Library::showDate() const {
    cout << "[当前日期: " << today.year << "/"
         << setfill('0') << setw(2) << today.month << "/"
         << setfill('0') << setw(2) << today.day << "]" << endl;
}

void Library::waitEnter() const {
    cout << endl << "按 Enter 继续...";
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
}

int Library::readInt(int lo, int hi) {
    int val;
    while (true) {
        cin >> val;
        if (cin.fail()) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "请输入数字: ";
            continue;
        }
        if (val >= lo && val <= hi) {
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            return val;
        }
        cout << "请输入 " << lo << "-" << hi << " 之间的数: ";
    }
}

Media* Library::findMedia(const string& id) {
    for (auto& m : mediaItems) {
        if (m->getId() == id) return m.get();
    }
    return nullptr;
}

User* Library::findUser(const string& id) {
    for (auto& u : users) {
        if (u->getId() == id) return u.get();
    }
    return nullptr;
}

void Library::cls() const {
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif
}

// ==================== 初始化演示数据 ====================
void Library::initData() {
    // 创建用户
    users.push_back(make_unique<Librarian>("admin", "王馆长", "123456"));
    users.push_back(make_unique<Reader>("u001", "张三", "123456"));
    users.push_back(make_unique<Reader>("u002", "李四", "123456"));

    // 创建馆藏
    mediaItems.push_back(make_unique<Book>(
        "B001", "C++ Primer Plus(第6版)", 2012,
        "Stephen Prata", "人民邮电出版社", "9787115279460", 936));
    mediaItems.push_back(make_unique<Book>(
        "B002", "Effective C++", 2011,
        "Scott Meyers", "电子工业出版社", "9787121123364", 320));
    mediaItems.push_back(make_unique<Book>(
        "B003", "设计模式：可复用面向对象软件的基础", 2000,
        "GoF", "机械工业出版社", "9787111075752", 416));
    mediaItems.push_back(make_unique<Magazine>(
        "M001", "程序员", 2026, 7, "CSDN传媒"));
    mediaItems.push_back(make_unique<Magazine>(
        "M002", "科幻世界", 2026, 6, "四川科幻世界杂志社"));
    mediaItems.push_back(make_unique<DVD>(
        "D001", "计算机科学导论", 2025,
        "David Malan", 180));

    // 预置一些借阅记录
    mediaItems[0]->setStatus(MediaStatus::BORROWED);
    dynamic_cast<Reader*>(users[1].get())->incBorrowed();
    borrowRecords.push_back(BorrowRecord(
        "B001", "u001",
        Date{2026, 6, 15}, Date{2026, 7, 15}));

    mediaItems[2]->setStatus(MediaStatus::BORROWED);
    dynamic_cast<Reader*>(users[2].get())->incBorrowed();
    borrowRecords.push_back(BorrowRecord(
        "B003", "u002",
        Date{2026, 6, 20}, Date{2026, 7, 20}));
}
