#ifndef LIBRARY_H
#define LIBRARY_H

#include <iostream>
#include <string>
#include <vector>
#include <memory>
// 头文件中不使用 using namespace std;

// ========== 前向声明 ==========
class Library;

// ========== 馆藏状态枚举 ==========
enum class MediaStatus { AVAILABLE, BORROWED, RESERVED, LOST };

// ========== Media 基类（馆藏） ==========
class Media {
protected:
    std::string id;          // 唯一编号
    std::string title;       // 题名
    int         year;        // 出版年份
    MediaStatus status;      // 当前状态

public:
    Media(std::string id, std::string title, int year);
    virtual ~Media();        // 虚析构函数

    virtual void display() const;               // 虚函数：展示信息
    virtual std::string getTypeName() const = 0; // 纯虚函数：获取类型名
    virtual int getLoanDays() const = 0;         // 纯虚函数：可借天数

    // getter
    std::string getId()     const { return id; }
    std::string getTitle()  const { return title; }
    MediaStatus getStatus() const { return status; }
    int getYear()           const { return year; }

    void setStatus(MediaStatus s) { status = s; }
};

// ========== Book 图书 ==========
class Book : public Media {
    std::string author;       // 作者
    std::string publisher;    // 出版社
    std::string isbn;         // ISBN
    int         pages;        // 页数

public:
    Book(std::string id, std::string title, int year,
         std::string author, std::string publisher,
         std::string isbn, int pages);

    void display() const override;
    std::string getTypeName() const override { return "图书"; }
    int getLoanDays() const override { return 30; }   // 图书可借30天

    std::string getAuthor() const { return author; }
    std::string getIsbn()   const { return isbn; }
};

// ========== Magazine 杂志 ==========
class Magazine : public Media {
    int         issueNumber;   // 期号
    std::string publisher;

public:
    Magazine(std::string id, std::string title, int year,
             int issue, std::string publisher);

    void display() const override;
    std::string getTypeName() const override { return "杂志"; }
    int getLoanDays() const override { return 7; }    // 杂志可借7天

    int getIssueNumber() const { return issueNumber; }
};

// ========== DVD 光盘 ==========
class DVD : public Media {
    std::string director;     // 导演
    int         duration;     // 时长(分钟)

public:
    DVD(std::string id, std::string title, int year,
        std::string director, int duration);

    void display() const override;
    std::string getTypeName() const override { return "DVD"; }
    int getLoanDays() const override { return 14; }   // DVD可借14天

    std::string getDirector() const { return director; }
};

// ========== User 基类（用户） ==========
class User {
protected:
    std::string userId;
    std::string name;
    std::string password;

public:
    User(std::string uid, std::string n, std::string pwd);
    virtual ~User();

    virtual bool canManageMedia() const = 0;   // 是否可管理馆藏
    virtual bool canManageUsers() const = 0;   // 是否可管理用户
    virtual std::string getRole() const = 0;   // 角色名
    virtual int getMaxBorrow() const = 0;      // 最大可借数

    std::string getId()   const { return userId; }
    std::string getName() const { return name; }
    bool checkPwd(std::string p) const { return password == p; }
};

// ========== Librarian 图书管理员 ==========
class Librarian : public User {
public:
    Librarian(std::string uid, std::string n, std::string pwd);

    bool canManageMedia() const override { return true; }
    bool canManageUsers() const override { return true; }
    std::string getRole() const override { return "图书管理员"; }
    int getMaxBorrow()  const override { return 20; }
};

// ========== Reader 读者 ==========
class Reader : public User {
    int borrowedCount;     // 当前已借数量

public:
    Reader(std::string uid, std::string n, std::string pwd);

    bool canManageMedia() const override { return false; }
    bool canManageUsers() const override { return false; }
    std::string getRole() const override { return "读者"; }
    int getMaxBorrow()  const override { return 5; }

    int  getBorrowedCount() const { return borrowedCount; }
    void incBorrowed() { borrowedCount++; }
    void decBorrowed() { borrowedCount--; }
};

// ========== 日期结构 & 工具函数 ==========
struct Date {
    int year, month, day;
};

bool operator<(const Date& a, const Date& b);
Date addDays(Date d, int days);
int  daysBetween(const Date& a, const Date& b);

// ========== BorrowRecord 借阅记录 ==========
class BorrowRecord {
    std::string mediaId;
    std::string userId;
    Date borrowDate;
    Date dueDate;
    bool returned;

public:
    BorrowRecord(std::string mid, std::string uid,
                 Date bDate, Date dDate);

    void display(const Media* media, const User* user) const;

    std::string getMediaId() const { return mediaId; }
    std::string getUserId() const { return userId; }
    Date getDueDate()     const { return dueDate; }
    bool isReturned()     const { return returned; }
    void setReturned()          { returned = true; }
    bool isOverdue(Date today) const;
    int  getOverdueDays(Date today) const;
};

// ========== Library 系统主控类 ==========
class Library {
    std::vector<std::unique_ptr<Media>> mediaItems;
    std::vector<std::unique_ptr<User>> users;
    std::vector<BorrowRecord> borrowRecords;
    User* currentUser;
    Date  today;
    bool  running;

public:
    Library();
    ~Library() = default;

    void run();

private:
    // 核心流程
    void login();
    void mainMenu();

    // 馆藏管理（管理员专属）
    void addMedia();
    void removeMedia();
    void searchMedia();
    void listAllMedia();

    // 借阅操作
    void borrowMedia();
    void returnMedia();
    void myBorrows();
    void allBorrows();    // 管理员查看全部借阅记录

    // 用户管理（管理员专属）
    void addUser();
    void listUsers();

    // 工具
    void waitEnter() const;
    int  readInt(int lo, int hi);
    void showDate() const;
    void advanceDate();
    void initData();
    void cls() const;

    // 查找辅助
    Media* findMedia(const std::string& id);
    User*  findUser(const std::string& id);
};

#endif
